import numpy as np
import pickle
import os

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
        self.prev_action = -1
        self.action = -1

    def update(self, env_state, action=None):
        if self.action is not None:
            self.prev_action = self.action
        self.action = action
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
        - 1維：prev_action [11]
        總共 12 維
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
            [self.prev_action]      # [11] 前一個動作
        ])
        
        return full_state
    
    def get_state_key(self, state):
        """將 state 轉換為可作為 Q 表 key 的 tuple"""
        return tuple(state.astype(int))

    

class StudentAgent:
    def __init__(self, checkpoint_path):
        self.action_dim = 6
        self.action = None
        self.obs = []
        self.unknown_state = []
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
            # Convert Q-table values from lists to numpy arrays for better performance
            self.q_table = {k: np.array(v) for k, v in checkpoint['q_table'].items()}
            # print(f"Loaded checkpoint from episode {checkpoint['episode']} with best reward: {checkpoint['best_reward']}")
        self.state_manager = State()
        print(f"Q table size: {len(self.q_table)}")    
        
    def reset(self):
        """重置 agent 狀態"""
        self.state_manager = State()
        
    def select_action(self, env_state):
        """根據環境狀態選擇動作"""
        # 更新內部狀態
        self.state_manager.update(env_state, action=self.action)
        
        # 獲取完整狀態表示
        full_state = self.state_manager.get_full_state(env_state)
        
        # 將狀態轉換為 Q table 的 key
        state_key = self.state_manager.get_state_key(full_state)

        # 根據 Q table 選擇最佳動作
        if state_key not in self.q_table:
            #print("NO")
            if state_key not in self.unknown_state:
                print(f"Unknown state: {state_key}")
                self.unknown_state.append(state_key)

            self.action = np.random.randint(self.action_dim)
            print(f"state_key: {state_key}, action: {self.action}")
            return self.action

        self.action = np.argmax(self.q_table[state_key])
       
        print(f"state_key: {state_key}, action: {self.action}")
        return self.action

agent = StudentAgent('checkpoints/checkpoint_30000.pt')
agent.reset()

def get_action(obs):
    return agent.select_action(obs)