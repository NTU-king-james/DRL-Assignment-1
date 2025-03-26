import gym
import numpy as np
from simple_custom_taxi_env import SimpleTaxiEnv
import os
import glob
import random
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from hard_env import HardTaxiEnv

# 修改後的 State 類別，包含 extended state 表示
class State:
    def __init__(self):
        self.visited_stations = [0, 0, 0, 0]  # 分別代表 R, G, Y, B 站點是否已拜訪
        self.take_status = False
        # 新增：可移動方向的狀態 (東南西北)
        self.can_move = [1, 1, 1, 1]
        # 新增：當前要訪問的站點索引
        self.current_target_station = 0
        self.at_station = False
        self.new_station = False  # 新增：是否到達新的站點
        self.passenger_pos = -1
        self.destination_pos = -1

    def update(self, env_state, action=None):

        taxi_row, taxi_col = env_state[0], env_state[1]
        taxi_pos = (taxi_row, taxi_col)
        
        # env_state[10:14] 分別代表東南西北方向是否有障礙物
        self.can_move = [
            not env_state[11],  # 南
            not env_state[10],  # 北
            not env_state[12],  # 東
            not env_state[13]   # 西
        ]
        
        # 更新 visited_stations：若 taxi 到達某站點，標記為已拜訪
        self.at_station = False
        self.new_station = False  # 重置 new_station 狀態
        idx = None
        for i, station_pos in enumerate([env_state[2:4], env_state[4:6], env_state[6:8], env_state[8:10]]):
            if taxi_pos == tuple(station_pos):
                if env_state[14]:
                    self.passenger_pos = i
                if env_state[15]:
                    self.destination_pos = i
                if self.visited_stations[i] == 0:  # 如果是新的站點
                    self.new_station = True
                self.visited_stations[i] = 1
                self.at_station = True
                idx = i
            
        # 處理 pickup 與 dropoff 動作
        if action == 4:  # pickup
            if env_state[-2] and self.at_station and not self.take_status:  # passenger_look
                self.take_status = True
                # 重置 visited_stations，只保留上車站點
                self.visited_stations = [0, 0, 0, 0]
                self.visited_stations[idx] = 1
        elif action == 5:  # dropoff
            self.take_status = False

        if not self.take_status:
            if self.passenger_pos != -1:
                self.current_target_station = self.passenger_pos
                return
        elif self.take_status:
            if self.destination_pos != -1:
                self.current_target_station = self.destination_pos
                return
       
        for j in range(4):
            if self.visited_stations[j] == 0:
                self.current_target_station = j
                break
    
    def get_full_state(self, env_state):
        """
        full state：
        - 2維：目標站點相對位置 [0-1]
        - 4維：可移動方向 [2-5]
        - 2維：乘客和目的地是否在附近 [6-7]
        - 1維：take_status [8]
        - 1維：at_station [9]
        - 1維：new_station [10]
        總共 11 維
        """
        # 計算目標站點相對於計程車的位置
        target_row, target_col = env_state[2 + self.current_target_station*2], env_state[3 + self.current_target_station*2]
        taxi_row, taxi_col = env_state[0], env_state[1]
        
        # 組合所有狀態
        rel_row = target_row - taxi_row
        rel_col = target_col - taxi_col
        full_state = np.concatenate([
            [rel_row, rel_col],    # [0-1] 目標站點相對位置
            self.can_move,         # [2-5] 可移動方向
            env_state[14:16],       # [6-7] 乘客和目的地狀態
            [self.take_status],     # [8] 載客狀態
            [self.at_station],      # [9] 是否在站點
            [self.new_station],      # [10] 是否到達新的站點
        ])
        
        return full_state
    
    def get_state_key(self, state):
        """將 state 轉換為可作為 Q 表 key 的 tuple"""
        return tuple(state.astype(int))


class Trainer:
    def __init__(self, fuel_limit=10000, num_episodes=120000, max_steps=5000, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999975, learning_rate=0.1):
        self.env = HardTaxiEnv(fuel_limit=fuel_limit)
        self.checkpoint_dir = "checkpoints"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.action_dim = 6  # 動作數量：南、北、東、西、pickup、dropoff
        
        # Q 表的 key 採用 extended state 的 tuple 表示，每個 state 對應 6 個動作的 Q 值
        self.q_table = defaultdict(lambda: np.zeros(self.action_dim))
        
        self.best_reward = float('-inf')
        self.episode_rewards = []
        self.episode_shaped_rewards = []
        self.done_episodes_history = []
        self.episode_numbers = []
        
        # 新增：追蹤前三個最佳模型
        self.top_3_models = []  # 格式: [(avg_reward, episode, q_table), ...]
        self.eval_window = 100  # 評估窗口大小
    
    def get_q_value(self, state_key, action):
        return self.q_table[state_key][action]
    
    def update_q_value(self, state_key, action, value):
        self.q_table[state_key][action] = value
    
    def select_action(self, state_key):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        return np.argmax(self.q_table[state_key])
    
    def calculate_shaped_reward(self, current_full_state, next_full_state, reward, done, action):
        """
        利用 potential-based reward shaping：
        定義潛力函數為 taxi 與目標站點 Manhattan 距離的負值，
        並引入 alpha 係數調整其影響力。
        """
        if done:
            return reward
        
        shaped_reward = reward - 0.4
        can_move = current_full_state[2:6]
        passenger_look = current_full_state[6]
        next_passenger_look = next_full_state[6]
        destination_look = current_full_state[7]
        next_destination_look = next_full_state[7]
        take_status = current_full_state[8]
        next_take_status = next_full_state[8]
        at_station = current_full_state[9]
        new_station = next_full_state[10]

        if action >= 4:
            if not at_station:
                shaped_reward -= 50
            if not take_status and next_take_status:
                shaped_reward += 50
            if take_status and not next_take_status:
                shaped_reward -= 100
            if not passenger_look and action == 4:
                shaped_reward -= 20
            if not destination_look and action == 5:
                shaped_reward -= 20
            return shaped_reward

        # 檢查是否訪問了新站點
        if new_station:
            shaped_reward += 30
            return shaped_reward
        
        if can_move[action] == 0:
            shaped_reward -= 30
        if at_station and not take_status and action != 4 and passenger_look:
            shaped_reward -= 10
        if at_station and take_status and action != 5 and destination_look:
            shaped_reward -= 10

        phi_current = abs(current_full_state[0]) + abs(current_full_state[1])
        phi_next = abs(next_full_state[0]) + abs(next_full_state[1])

        if phi_next - phi_current < 0:
            shaped_reward += 1
        if phi_next - phi_current > 0:
            shaped_reward -= 5

        if not take_status and not passenger_look and next_passenger_look:
            shaped_reward += 30
        if not take_status and passenger_look and not next_passenger_look:
            shaped_reward -= 30
        if take_status and not destination_look and next_destination_look:
            shaped_reward += 30
        if take_status and destination_look and not next_destination_look:
            shaped_reward -= 30
            
        return shaped_reward  # 確保所有情況都有返回值
    
    def plot_training_progress(self):
        """繪製訓練進度的圖表"""
        plt.figure(figsize=(12, 5))
        
        # 繪製reward變化
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_numbers, self.episode_rewards, label='Episode Reward')
        plt.title('Episode Rewards Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        
        # 繪製完成episode的變化
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_numbers, self.done_episodes_history, label='Done Episodes')
        plt.title('Completed Episodes Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Done Episodes')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()
    
    def save_checkpoint(self, episode):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{episode}.pt")
        serializable_q_table = {}
        for state_key, q_values in self.q_table.items():
            serializable_q_table[state_key] = q_values.tolist()
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'episode': episode,
                'q_table': serializable_q_table,
                'best_reward': self.best_reward,
            }, f)
        self.clean_old_checkpoints()


    def update_top_3_models(self, episode):
        # 只有在 90000 epoch 之後才開始更新 top 3 模型
        if episode < 100000:
            return

        if len(self.episode_rewards) < 100:  # 確保至少有100個 episode 的數據
            return

        # 計算最近100個 episode 的平均獎勵
        recent_avg_reward = np.mean(self.episode_rewards[-100:])
        
        # 準備當前模型
        current_model = (recent_avg_reward, episode, self.q_table.copy())
        
        # 更新前三個最佳模型
        if len(self.top_3_models) < 3:
            self.top_3_models.append(current_model)
            self.top_3_models.sort(key=lambda x: x[0], reverse=True)
        else:
            # 如果當前模型比最差的好，則替換
            if recent_avg_reward > self.top_3_models[-1][0]:
                # 刪除舊檔案：取出最差模型的 episode
                worst_ep = self.top_3_models[-1][1]
                worst_file_pattern = os.path.join(self.checkpoint_dir, f"top_*_model_episode_{worst_ep}.pt")
                for file in glob.glob(worst_file_pattern):
                    os.remove(file)
                self.top_3_models.pop()
                self.top_3_models.append(current_model)
                self.top_3_models.sort(key=lambda x: x[0], reverse=True)
        
        # 儲存當前 top 3 模型（以固定命名覆蓋更新）
        for i, (avg_reward, ep, q_table) in enumerate(self.top_3_models):
            model_path = os.path.join(self.checkpoint_dir, f"top_{i+1}_model_episode_{ep}.pt")
            serializable_q_table = {}
            for state_key, q_values in q_table.items():
                serializable_q_table[state_key] = q_values.tolist()
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'episode': ep,
                    'q_table': serializable_q_table,
                    'avg_reward': avg_reward,
                }, f)
        
        # 刪除所有不在目前 top 3 模型中的 top 模型檔案
        self.clean_old_top_model_checkpoints()


    def clean_old_top_model_checkpoints(self):
        # 找出所有符合 top 模型檔案格式的檔案
        top_files = glob.glob(os.path.join(self.checkpoint_dir, "top_*_model_episode_*.pt"))
        # 取得目前 top 3 模型的 episode 集合
        valid_episodes = {model[1] for model in self.top_3_models}
        for file in top_files:
            # 從檔名中解析 episode 數字，檔名格式例如 "top_1_model_episode_12345.pt"
            try:
                ep_num = int(file.split('_')[-1].split('.')[0])
            except ValueError:
                continue  # 若解析失敗就跳過
            if ep_num not in valid_episodes:
                os.remove(file)


    def clean_old_checkpoints(self):
        # 清除一般 checkpoint 檔案，只保留最新的 3 個
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pt"))
        # 使用 episode 數字進行排序
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                os.remove(old_checkpoint)

    
    def train(self):
        done_episodes = 0
        for episode in range(self.num_episodes):
            env_state, _ = self.env.reset()
            state_manager = State()
            state_manager.update(env_state)
            
            current_full_state = state_manager.get_full_state(env_state)
            current_state_key = state_manager.get_state_key(current_full_state)
            
            episode_reward = 0
            episode_shaped_rewards = 0
            
            for step in range(self.max_steps):
                action = self.select_action(current_state_key)
                next_env_state, reward, done, _ = self.env.step(action)
                state_manager.update(next_env_state, action)
                
                next_full_state = state_manager.get_full_state(next_env_state)
                next_state_key = state_manager.get_state_key(next_full_state)

                shaped_reward = self.calculate_shaped_reward(current_full_state, next_full_state, reward, done, action)
                
                current_q = self.get_q_value(current_state_key, action)
                next_max_q = max(self.get_q_value(next_state_key, a) for a in range(self.action_dim))
                new_q = current_q + self.learning_rate * (shaped_reward + self.gamma * next_max_q - current_q)
                self.update_q_value(current_state_key, action, new_q)
                
                episode_reward += reward
                episode_shaped_rewards += shaped_reward
                take_status = current_full_state[8]
                next_take_status = next_full_state[8]
                if done:
                    done_episodes += 1
                    break
                if take_status and not next_take_status:
                    break
                current_state_key = next_state_key
                current_full_state = next_full_state
            
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.episode_rewards.append(episode_reward)
            self.episode_shaped_rewards.append(episode_shaped_rewards)
            self.done_episodes_history.append(done_episodes)
            self.episode_numbers.append(episode + 1)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_shaped_reward = np.mean(self.episode_shaped_rewards[-100:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}, Average Shaped Reward: {avg_shaped_reward:.2f}, Done Episodes: {done_episodes}")
                done_episodes = 0
                
                # 在 90000 epoch 後開始保存和更新前三個最佳模型
                if episode + 1 >= 100000:
                    self.save_checkpoint(episode + 1)
                    self.update_top_3_models(episode + 1)
                    
                    # 打印前三個最佳模型的資訊
                    print("\nTop 3 Models:")
                    for i, (avg_reward, ep, _) in enumerate(self.top_3_models):
                        print(f"Rank {i+1}: Episode {ep}, Average Reward: {avg_reward:.2f}")
            
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                print(f"New best reward: {self.best_reward}")
        
        print(f"Training finished. Best reward: {self.best_reward}")
        self.plot_training_progress()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
