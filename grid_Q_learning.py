import numpy as np
import matplotlib.pyplot as plt



class GridWorldEnv:
    """
    4x4 网格世界环境
    - 状态: 0-15（每个格子对应一个状态，从左到右、从上到下编号）
    - 动作: 0=右, 1=左, 2=上, 3=下
    - 目标: 到达右下角(3,3)即状态15，获得奖励1
    """
    def __init__(self):
        self.grid_size = 4
        self.target = (3, 3)  # 目标位置
        self.loc = [0, 0]  # 当前位置 [行, 列]

    def get_state(self):
        """
        将二维坐标位置转换为一维状态编号 (0-15)
        
        转换公式: state = row * grid_size + col
        示例: 位置(0,0)→状态0, 位置(3,3)→状态15
        
        Returns:
            int: 状态编号 (0-15)
        """
        return self.loc[0] * self.grid_size + self.loc[1]

    def step(self, action):
        """
        执行智能体动作，更新环境状态并返回结果
        
        Args:
            action (int): 动作编号，0=右, 1=左, 2=上, 3=下
            
        Returns:
            tuple: (next_state, reward, done)
                next_state: 执行动作后的状态编号
                reward: 获得的奖励值（到达目标+1，否则-0.1）
                done: 是否到达终止状态（目标位置）
        """
        # 根据动作更新位置坐标，并进行边界检查防止越界
        if action == 0:  # 向右移动：列坐标+1
            self.loc[1] = min(self.grid_size - 1, self.loc[1] + 1)  # 不超过右边界
        elif action == 1:  # 向左移动：列坐标-1
            self.loc[1] = max(0, self.loc[1] - 1)  # 不超过左边界
        elif action == 2:  # 向上移动：行坐标-1
            self.loc[0] = max(0, self.loc[0] - 1)  # 不超过上边界
        elif action == 3:  # 向下移动：行坐标+1
            self.loc[0] = min(self.grid_size - 1, self.loc[0] + 1)  # 不超过下边界

        # 计算执行动作后的新状态编号
        next_state = self.get_state()

        # 判断是否到达目标位置(3,3)，并计算奖励
        if self.loc[0] == self.target[0] and self.loc[1] == self.target[1]:
            reward = 5.0  # 到达目标，获得正奖励
            done = True   # 标记任务完成
        else:
            reward = -0.1  # 未到达目标，给予小惩罚，鼓励寻找最短路径
            done = False   # 任务未完成

        return next_state, reward, done

    def reset(self):
        """重置位置到起点(0,0)"""
        self.loc = [0, 0]
        return self.get_state()
class Qlearning:
    """
    Q-learning 强化学习智能体
    
    Q-learning 是一种无模型强化学习算法，通过学习状态-动作价值函数(Q值)来指导决策。
    核心思想是通过贝尔曼方程迭代更新Q表，最终学习到最优策略。
    """
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=1.0):
        """
        初始化Q-learning智能体
        
        Args:
            states (int): 状态空间大小（状态总数）
            actions (int): 动作空间大小（动作总数）
            alpha (float, optional): 学习率，控制Q值更新幅度，范围(0,1]，默认0.1
            gamma (float, optional): 折扣因子，控制未来奖励的权重，范围[0,1]，默认0.9
            epsilon (float, optional): epsilon-greedy策略的初始探索率，范围[0,1]，默认1.0
        """
        self.states = states          # 状态空间大小
        self.actions = actions        # 动作空间大小
        self.alpha = alpha            # 学习率 (learning rate)
        self.gamma = gamma            # 折扣因子 (discount factor)
        self.epsilon = epsilon        # 探索率 (exploration rate)

        # 初始化Q表：states x actions 的零矩阵
        # Q[state, action] 表示在状态state下执行动作action的期望累计奖励
        self.Q = np.zeros((states, actions))

    def choose_action(self, state):
        """
        使用epsilon-greedy策略选择动作
        
        以epsilon概率随机选择动作（探索），以1-epsilon概率选择Q值最大的动作（利用）
        
        Args:
            state (int): 当前状态编号
            
        Returns:
            int: 选择的动作编号
        """
        if np.random.rand() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.actions)
        else:
            # 利用：选择当前Q值最大的动作
            return np.argmax(self.Q[state])

    def update(self, state, action, next_state, reward):
        """
        使用Q-learning算法更新Q值
        
        Q-learning 更新公式:
        Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
        
        Args:
            state (int): 当前状态
            action (int): 执行的动作
            next_state (int): 执行动作后的状态
            reward (float): 获得的奖励
        """
        # 找到下一状态的最优动作
        best_next_action = np.argmax(self.Q[next_state])
        # 计算时序差分目标值
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        # 计算时序差分误差
        td_error = td_target - self.Q[state, action]
        # 更新Q值
        self.Q[state, action] += self.alpha * td_error

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """
        指数衰减探索率epsilon
        
        随着训练进行，逐渐降低探索率，增加利用已学习知识的概率
        
        Args:
            decay_rate (float, optional): 衰减系数，默认0.995
            min_epsilon (float, optional): 最小探索率，默认0.01
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

# ==================== 训练参数设置 ====================
train_episodes = 500    # 训练总轮数
env = GridWorldEnv()    # 创建网格世界环境实例
episode_rewards = []    # 记录每轮奖励的列表
agent = Qlearning(16, 4)  # 创建Q-learning智能体（16个状态，4个动作）

# ==================== 主训练循环 ====================
print("开始训练 Q-learning 智能体...")
for episode in range(train_episodes):
    state = env.reset()     # 重置环境到初始状态(0,0)
    total_reward = 0        # 累计奖励初始化
    done = False            # 终止标志初始化

    # 单轮训练循环（直到到达目标或达到最大步数）
    while not done:
        action = agent.choose_action(state)          # 根据当前状态选择动作
        next_state, reward, done = env.step(action)  # 执行动作，获取下一状态和奖励
        agent.update(state, action, next_state, reward)  # 更新Q值
        state = next_state   # 更新当前状态
        total_reward += reward  # 累计奖励
    
    agent.decay_epsilon()   # 降低探索率
    episode_rewards.append(total_reward)  # 记录本轮奖励

    # 每50轮打印一次训练进度
    if (episode + 1) % 50 == 0:
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        print(f"Episode {episode + 1}/{train_episodes}, Average Reward: {avg_reward:.2f}")

print("训练完成!")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图像
plt.figure(figsize=(10, 5))

# 绘制原始奖励曲线
plt.plot(range(train_episodes), episode_rewards, label='每轮奖励', alpha=0.6, color='#1f77b4')

# 计算并绘制滑动平均曲线（更清晰展示趋势）
window_size = 10
if len(episode_rewards) >= window_size:
    smoothed_rewards = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, train_episodes), smoothed_rewards, 
             'r-', linewidth=2, label='滑动平均奖励')

# 添加标题和标签
plt.title('Q-learning 训练奖励曲线', fontsize=14, fontweight='bold')
plt.xlabel('训练轮数 (Episode)', fontsize=12)
plt.ylabel('总奖励 (Total Reward)', fontsize=12)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例
plt.legend(fontsize=12)

# 添加奖励范围标注
max_reward = max(episode_rewards)
min_reward = min(episode_rewards)
plt.annotate(f'最高奖励: {max_reward:.2f}', xy=(train_episodes*0.05, max_reward*0.95),
             xycoords='data', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.8))
plt.annotate(f'最低奖励: {min_reward:.2f}', xy=(train_episodes*0.05, min_reward*1.05),
             xycoords='data', bbox=dict(boxstyle='round,pad=0.3', fc='pink', alpha=0.8))

plt.show()
