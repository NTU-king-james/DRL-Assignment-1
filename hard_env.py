import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random
# This environment allows you to verify whether your program runs correctly during testing, 
# as it follows the same observation format from `env.reset()` and `env.step()`. 
# However, keep in mind that this is just a simplified environment. 
# The full specifications for the real testing environment can be found in the provided spec.
# 
# You are free to modify this file to better match the real environment and train your own agent. 
# Good luck!


class HardTaxiEnv():
    def __init__(self, grid_size=5, fuel_limit=1000):
        """
        Custom Taxi environment supporting different grid sizes.
        """
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False
        
        self.passenger_loc = None
        self.stations = None
        self.obstacles = set()  # Set of obstacle positions
        self.destination = None

    def is_valid_obstacle(self, pos, existing_obstacles):
        """檢查障礙物位置是否有效（不相連且不阻擋路徑）"""
        if not (0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size):
            return False
        # 檢查是否與現有障礙物相鄰
        for obs in existing_obstacles:
            if abs(pos[0] - obs[0]) + abs(pos[1] - obs[1]) <= 1:
                return False
        return True

    def is_path_exists(self, start, end, obstacles):
        """使用 BFS 檢查兩點之間是否存在路徑"""
        from collections import deque
        queue = deque([start])
        visited = {start}
        
        while queue:
            current = queue.popleft()
            if current == end:
                return True
                
            # 檢查四個方向
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + dx, current[1] + dy)
                if (0 <= next_pos[0] < self.grid_size and 
                    0 <= next_pos[1] < self.grid_size and 
                    next_pos not in obstacles and 
                    next_pos not in visited):
                    queue.append(next_pos)
                    visited.add(next_pos)
        return False

    def reset(self):
        """Reset the environment, ensuring Taxi, passenger, and destination are not overlapping obstacles"""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
        
        self.grid_size = random.randint(5, 10)  # Random size between 5 and 10
        
        # 生成隨機站點位置，確保不相連
        def is_valid_station(pos, existing_stations):
            if not (0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size):
                return False
            # 檢查是否與現有站點相鄰
            for station in existing_stations:
                if abs(pos[0] - station[0]) + abs(pos[1] - station[1]) <= 1:
                    return False
            return True
        
        # 生成四個不相連的站點
        self.stations = []
        max_attempts = 100  # 防止無限循環
        attempts = 0
        
        while len(self.stations) < 4 and attempts < max_attempts:
            pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if is_valid_station(pos, self.stations):
                self.stations.append(pos)
            attempts += 1
        
        # 如果無法生成足夠的站點，使用預設位置
        if len(self.stations) < 4:
            self.stations = [(0, 0), (0, self.grid_size - 1), 
                           (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
        
        # 生成隨機障礙物
        self.obstacles = set()
        num_obstacles = random.randint(0, self.grid_size)  # 隨機障礙物數量
        attempts = 0
        max_attempts = 1000  # 防止無限循環

        while len(self.obstacles) < num_obstacles and attempts < max_attempts:
            pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if pos not in self.stations and self.is_valid_obstacle(pos, self.obstacles):
                # 模擬加入新障礙物
                temp_obstacles = self.obstacles | {pos}

                # 檢查整個 grid 裡所有自由格子是否至少有一個鄰近格子可以通行
                isolated = False
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        if (i, j) not in temp_obstacles:
                            # 取得上下左右鄰居（在 grid 內的）
                            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                            valid_neighbors = [(ni, nj) for (ni, nj) in neighbors if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size]
                            # 如果所有鄰居都是障礙物，就表示 (i,j) 被封住
                            if all(neighbor in temp_obstacles for neighbor in valid_neighbors):
                                isolated = True
                                break
                    if isolated:
                        break

                # 若沒有產生任何孤立的自由格子，再檢查站點之間的連通性
                if not isolated:
                    valid = True
                    for i in range(len(self.stations)):
                        for j in range(i + 1, len(self.stations)):
                            if not self.is_path_exists(self.stations[i], self.stations[j], temp_obstacles):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        self.obstacles.add(pos)
            attempts += 1


        # 生成計程車、乘客和目的地位置
        available_positions = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
            if (x, y) not in self.stations and (x, y) not in self.obstacles
        ]

        self.taxi_pos = random.choice(available_positions)
        self.passenger_loc = random.choice([pos for pos in self.stations])
        possible_destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_destinations)
        
        return self.get_state(), {}

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        if action == 0 :  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1
        
        
        if action in [0, 1, 2, 3]:  # Only movement actions should be checked
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -=5
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        else:
            if action == 4:  # PICKUP
                if self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos  
                else:
                    reward = -10  
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 50
                        return self.get_state(), reward -0.1, True, {}
                    else:
                        reward -=10
                    self.passenger_picked_up = False
                    self.passenger_loc = self.taxi_pos
                else:
                    reward -=10
                    
        reward -= 0.1  

        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward -10, True, {}

        

        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state."""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination
        
        obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row , taxi_col-1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int( (taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
       
        destination_loc_north = int( (taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int( (taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int( (taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int( (taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle  = int( (taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle

        
        state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state
    def render_env(self, taxi_pos,   action=None, step=None, fuel=None):
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        
        '''
        # Place passenger
        py, px = passenger_pos
        if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
            grid[py][px] = 'P'
        '''
        
        
        grid[0][0]='R'
        grid[0][self.grid_size-1]='G'
        grid[self.grid_size-1][0]='Y'
        grid[self.grid_size-1][self.grid_size-1]='B'
        '''
        # Place destination
        dy, dx = destination_pos
        if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
            grid[dy][dx] = 'D'
        '''
        # Place taxi
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = '🚖'

        # Print step info
        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        #print(f"Passenger Position: ({px}, {py}) {'(In Taxi)' if (px, py) == (tx, ty) else ''}")
        #print(f"Destination: ({dx}, {dy})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"


def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = HardTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    # stations = [(0, 0), (0, 4), (4, 0), (4,4)]
    
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    if render:
        env.render_env((taxi_row, taxi_col),
                       action=None, step=step_count, fuel=env.current_fuel)
        time.sleep(0.5)
    while not done:
        
        
        action = student_agent.get_action(obs)

        obs, reward, done, _ = env.step(action)
        print('obs=',obs)
        total_reward += reward
        step_count += 1

        taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs

        if render:
            env.render_env((taxi_row, taxi_col),
                           action=action, step=step_count, fuel=env.current_fuel)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    env_config = {
        "fuel_limit": 5000
    }
    
    agent_score = run_agent("student_agent.py", env_config, render=False)
    print(f"Final Score: {agent_score}")