"""
微电网调度 DQN（光伏 + 电池 + 电网交互）
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'


# ==================== 1. 微电网环境 ====================
class MicrogridEnv:
    """
    状态: [当前时刻(h), 光伏出力(kW), 负载需求(kW), 电价(元/kWh), 电池SOC(%)]
    动作: 0~9，对应电池净放电功率: action * 5 - 20  (范围 -20 kW ~ +25 kW)
          负值 = 电池放电，正值 = 电池充电
    实际调度中，电网自动平衡剩余缺口
    """

    def __init__(self):
        self.hour = 0
        self.soc = 50.0
        self.soc_min = 10.0
        self.soc_max = 90.0
        self.battery_capacity = 100.0  # kWh
        self.max_power = 25.0  # 最大充放电功率 kW

        # 固定一天的辐照度曲线（kW，峰值30kW的光伏系统）
        self.solar_curve = np.array([
            0, 0, 0, 0, 0, 0, 2, 6, 12, 18, 23, 27, 29, 30, 28, 25,
            20, 14, 8, 4, 1, 0, 0, 0
        ], dtype=np.float32)

        # 固定一天的负载曲线（kW）
        self.load_curve = np.array([
            8, 6, 5, 4, 5, 7, 10, 12, 11, 10, 10, 12, 14, 13, 12, 11,
            14, 18, 20, 17, 13, 10, 8, 6
        ], dtype=np.float32)

        # 电价曲线（元/kWh，中午光伏多价格低，早晚高峰价格高）
        self.price_curve = np.array([
            0.30, 0.25, 0.20, 0.18, 0.15, 0.18, 0.25, 0.35,
            0.30, 0.25, 0.20, 0.15, 0.12, 0.15, 0.20, 0.30,
            0.50, 0.70, 0.80, 0.70, 0.50, 0.35, 0.30, 0.30
        ], dtype=np.float32)

    def reset(self):
        self.hour = 0
        self.soc = 50.0
        return self._get_state()

    def _get_state(self):
        h = self.hour / 24.0
        solar = self.solar_curve[self.hour] / 30.0
        load = self.load_curve[self.hour] / 20.0
        price = self.price_curve[self.hour]
        soc = self.soc / 100.0
        return np.array([h, solar, load, price, soc], dtype=np.float32)

    def step(self, action):
        """
        action: 0~9 整数
        battery_power = action * 5 - 20  (范围 -20 ~ +25 kW)
        """
        # 解析动作 → 电池功率
        battery_power = action * 5.0 - 20.0  # -20 到 +25

        solar_power = self.solar_curve[self.hour]
        load_power = self.load_curve[self.hour]
        price = self.price_curve[self.hour]

        # 限制电池功率在物理范围内
        if battery_power > 0:  # 充电
            max_charge = min(self.max_power,
                             (self.soc_max - self.soc) / 100.0 * self.battery_capacity)
            battery_power = min(battery_power, max_charge)
        else:  # 放电
            max_discharge = min(self.max_power,
                                (self.soc - self.soc_min) / 100.0 * self.battery_capacity)
            battery_power = max(battery_power, -max_discharge)

        # 更新 SOC
        self.soc += battery_power / self.battery_capacity * 100.0
        self.soc = np.clip(self.soc, self.soc_min, self.soc_max)

        # 电网交互功率 = 负载 - 光伏 - 电池放电(+)
        grid_power = load_power - solar_power - (-battery_power)

        # 奖励 = 光伏自用节省的钱 - 电网购电成本 + 电网售电收入
        # 简化：正向现金流
        reward = -price * grid_power  # 电网购电为负奖励，售电为正奖励

        self.hour = (self.hour + 1) % 24
        done = (self.hour == 0)

        return self._get_state(), reward, done, {}

    def get_optimal_revenue_upper_bound(self):
        """理论最大收益（完美预测下）"""
        soc = 50.0
        total = 0
        for h in range(24):
            solar = self.solar_curve[h]
            load = self.load_curve[h]
            price = self.price_curve[h]
            net = solar - load

            if net > 0:  # 有余电
                charge = min(net, (90 - soc) / 100.0 * self.battery_capacity)
                soc += charge / self.battery_capacity * 100
                sold = net - charge
                total += price * sold
            else:  # 不够
                discharge = min(-net, (soc - 10) / 100.0 * self.battery_capacity)
                soc -= discharge / self.battery_capacity * 100
                bought = -net - discharge
                total -= price * bought
        return total


# ==================== 2. DQN 网络 ====================
class DQN(nn.Module):
    def __init__(self, state_dim=5, action_dim=10, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ==================== 3. 经验回放 ====================
class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, acts, rews, next_states, dones = zip(*batch)
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(acts, dtype=torch.long),
                torch.tensor(rews, dtype=torch.float32),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)


# ==================== 4. 训练 ====================
env = MicrogridEnv()
state_dim = 5
action_dim = 10

online_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(online_net.state_dict())

optimizer = optim.Adam(online_net.parameters(), lr=0.0005)
criterion = nn.MSELoss()
memory = ReplayBuffer(5000)

EPISODES = 400
BATCH_SIZE = 128
GAMMA = 0.95
EPSILON_START = 0.9
EPSILON_END = 0.02
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10

epsilon = EPSILON_START
episode_rewards = []

print("=" * 60)
print("   微电网调度 DQN 训练")
print("=" * 60)
print(f"状态维度: {state_dim} (h, solar, load, price, soc)")
print(f"动作维度: {action_dim} (电池功率 -20 ~ +25 kW)")

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for _ in range(24):
        if random.random() < epsilon:
            action = random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = online_net(state_t).argmax().item()

        next_state, reward, done, _ = env.step(action)
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(memory) > BATCH_SIZE:
            s, a, r, ns, d = memory.sample(BATCH_SIZE)
            curr_q = online_net(s).gather(1, a.unsqueeze(1)).squeeze()
            with torch.no_grad():
                target_q = r + GAMMA * target_net(ns).max(1)[0] * (1 - d)
            loss = criterion(curr_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    episode_rewards.append(total_reward)
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if (episode + 1) % TARGET_UPDATE == 0:
        target_net.load_state_dict(online_net.state_dict())

    if (episode + 1) % 50 == 0:
        avg_r = np.mean(episode_rewards[-50:])
        print(f"回合 {episode + 1:3d}/{EPISODES} | 平均奖励: {avg_r:.2f} | ε: {epsilon:.3f}")

# ==================== 5. 测试策略 ====================
print("\n" + "=" * 60)
print("   测试最优策略")
print("=" * 60)

state = env.reset()
online_net.eval()

soc_history = [state[4] * 100]
price_history = [state[3]]
solar_history = [state[1] * 30]
load_history = [state[2] * 20]
grid_history = []
battery_history = []
reward_history = []

for _ in range(24):
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = online_net(state_t).argmax().item()

    battery_power = action * 5.0 - 20.0
    battery_history.append(battery_power)

    next_state, reward, done, _ = env.step(action)

    soc_history.append(next_state[4] * 100)
    price_history.append(next_state[3])
    solar_history.append(next_state[1] * 30)
    load_history.append(next_state[2] * 20)

    # 计算电网功率
    solar = solar_history[-2] if len(solar_history) > 1 else solar_history[0]
    load = load_history[-2] if len(load_history) > 1 else load_history[0]
    grid_power = load - solar - (-battery_power)
    grid_history.append(grid_power)

    reward_history.append(reward)
    state = next_state

total_reward = sum(reward_history)
opt_upper = env.get_optimal_revenue_upper_bound()
print(f"DQN 24h 收益: {total_reward:.2f} 元")
print(f"理论上界:      {opt_upper:.2f} 元")
print(f"达到理论上界的 {total_reward / opt_upper * 100:.1f}%")

# ==================== 6. 可视化 ====================
fig, axes = plt.subplots(3, 2, figsize=(18, 12))

# 图1：光伏 + 负载
ax1 = axes[0, 0]
hours = np.arange(24)
ax1.fill_between(hours, 0, solar_history[:24], color='orange', alpha=0.4, label='光伏出力')
ax1.plot(hours, load_history[:24], 'b-', linewidth=2, label='负载需求')
ax1.set_xlabel("小时")
ax1.set_ylabel("功率 (kW)")
ax1.set_title("光伏与负载")
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.5)

# 图2：电池充放电 + SOC
ax2 = axes[0, 1]
colors_bat = ['green' if p > 0 else 'red' for p in battery_history]
ax2.bar(hours, battery_history, color=colors_bat, alpha=0.6, label='电池充(+)/放(-)')
ax2_twin = ax2.twinx()
ax2_twin.plot(hours, soc_history[:24], 'k-o', linewidth=2, markersize=5, label='SOC')
ax2.set_xlabel("小时")
ax2.set_ylabel("电池功率 (kW)")
ax2_twin.set_ylabel("SOC (%)")
ax2.set_title("电池调度策略")
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax2.grid(True, linestyle=':', alpha=0.5)

# 图3：电网交互
ax3 = axes[1, 0]
colors_grid = ['blue' if p > 0 else 'green' for p in grid_history]
ax3.bar(hours, grid_history, color=colors_grid, alpha=0.6, label='购电(+)/售电(-)')
ax3.axhline(y=0, color='black', linewidth=1)
ax3.set_xlabel("小时")
ax3.set_ylabel("电网交互功率 (kW)")
ax3.set_title("电网交互 (正=购电, 负=售电)")
ax3.grid(True, linestyle=':', alpha=0.5)

# 图4：电价 + 动作热图
ax4 = axes[1, 1]
ax4.plot(hours, price_history[:24], 'r-', linewidth=2, markersize=6, marker='o', label='电价')
ax4.set_xlabel("小时")
ax4.set_ylabel("电价 (元/kWh)")
ax4.set_title("电价曲线")
ax4.legend()
ax4.grid(True, linestyle=':', alpha=0.5)

# 图5：累计收益
ax5 = axes[2, 0]
cum_reward = np.cumsum(reward_history)
ax5.plot(hours, cum_reward, 'steelblue', linewidth=2.5)
ax5.axhline(y=opt_upper, color='green', linestyle='--', linewidth=2, label=f'理论上界 {opt_upper:.1f}元')
ax5.set_xlabel("小时")
ax5.set_ylabel("累计收益 (元)")
ax5.set_title("24h 累计收益")
ax5.legend()
ax5.grid(True, linestyle=':', alpha=0.5)

# 图6：训练曲线
ax6 = axes[2, 1]
ax6.plot(episode_rewards, 'steelblue', linewidth=1, alpha=0.5)
smooth = np.convolve(episode_rewards, np.ones(40) / 40, mode='valid')
ax6.plot(range(39, len(episode_rewards)), smooth, 'red', linewidth=2, label='40回合移动平均')
ax6.set_xlabel("训练回合")
ax6.set_ylabel("总奖励")
ax6.set_title("DQN 训练曲线")
ax6.legend()
ax6.grid(True, linestyle=':', alpha=0.5)

plt.suptitle("微电网智能调度 (DQN)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("microgrid_dqn.png", dpi=200)
plt.show()