import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import logging

""" Four rooms. The goal is either in the 3rd room, or in a hallway adjacent to it
"""
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
NUM_ROOMS =4

class Fourrooms(gym.Env):
    metadata = {'render.modes': ['human']}
        
    def __init__(self):
        self.grid = np.zeros([11,11])
        self.goal_positions = [[6,8],[8,8]]
        self.actions = {0:[-1,0]      #N
        ,1:[0,1],2:[1,0],3:[0,-1]
        }
        #self.action_space = spaces.discrete(len(self.actions)*2)
        self.observation_space = spaces.Box(low = -5, high = 10, shape = self.grid.shape)

        self.grid[5,0:1] -= 0.1
        self.grid[5,2:5] -= 0.1
        self.grid[6,6:8] -= 0.1
        self.grid[6,9:11]-= 0.1
        
        self.grid[0:2,5]        -= 0.1
        self.grid[3:9,5]        -= 0.1
        self.grid[10:11,5] -= 0.1

        self.grid[6,8] += 1                   #GOAL POSITION

        self.rooms = {                                                  #start position and size of each room
                        1 : [[0,0], [5,5]],
                        2 : [[0,6], [6,5]],
                        3 : [[7,6], [4,5]],
                        4 : [[6,0], [5,5]]     
                        }
        self.doorways = { 
                          1: [[5,1],[2,5]],
                          2 : [[2,5],[6,8]],
                          3: [[6,8],[9,5]],
                          4: [[9,5],[5,1]]      }

        
    def get_start_position(self):
        start_room = 1
        st_pos,dim = self.rooms[start_room]
        x_range = list(range(st_pos[0],st_pos[0]+dim[0]))
        y_range = list(range(st_pos[1],st_pos[1]+dim[1]))
        start_pos = [np.random.choice(x_range),np.random.choice(y_range)]
        return start_pos 

    def get_room(self,state):
        r1 = self.rooms[1]
        r2 = self.rooms[2]
        r3 = self.rooms[3]
        r4 = self.rooms[4]
        if (state[0]>=r1[0][0] and state[0]<=r1[0][0] + r1[1][0] and state[1]>=r1[0][1] and state[1]<=r1[0][1] + r1[1][1] ):
            return 1
        elif (state[0]>=r2[0][0] and state[0]<=r2[0][0] + r2[1][0] and state[1]>=r2[0][1] and state[1]<=r2[0][1] + r2[1][1]):
            return 2
        elif (state[0]>=r3[0][0] and state[0]<=r3[0][0] + r3[1][0] and state[1]>=r3[0][1] and state[1]<=r3[0][1] + r3[1][1]):
            return 3
        elif (state[0]>=r4[0][0] and state[0]<=r4[0][0] + r4[1][0] and state[1]>=r4[0][1] and state[1]<=r4[0][1] + r4[1][1]):
            return 4
        else:
            return 0 

    def get_doorways(self,state):
        room_num = self.get_room(state)
        if room_num!= 0 :
            drwys = self.doorways[room_num]
        return drwys

    def get_reward(self,position):                                 
        reward = self.grid[position[0],position[1]]
        return reward
 #function to check if in hallway
    def in_doorway(self,state):
        d1,d2 = self.doorways[1]
        d3,d4 = self.doorways[2]

        if state == d1:
            return d1,True
        elif state ==d2:
            return d2,True
        elif state ==d3:
            return d3,True
        elif state == d4:
            return d4,True
        else:
            return state,False
    # def coord2ind(coord):
    #     [r,c] = coord
    #     return r*11 + c

    def step(self,state,action,target_doorway):
        #beta = np.zeros([11,11])
        room = self.get_room(state)
        x1 = self.rooms[room][0][0]                                   
        x2 = self.rooms[room][0][0]+self.rooms[room][1][0]-1
        y1 = self.rooms[room][0][1]
        y2 = self.rooms[room][0][1]+self.rooms[room][1][1]-1
        state,flag = self.in_doorway(state)
        if flag:                                                         #in hallway
            x,y = state
            x_ = self.actions[action][0]
            y_ = self.actions[action][1]
            rew = self.get_reward([x+x_,y+y_])
            if (rew==-0.1):
                next_state = state
            else:
                next_state = [x+x_,y+y_]
            reward = rew
            terminate = False
            done = False
        else:                                                            #  normal state
            drwys = self.get_doorways(state)
            d1,d2 = drwys
            p = state[0]+self.actions[action][0]
            q = state[1]+self.actions[action][1]
            next_state = [p,q]
            if ([p,q]==d1 or [p,q]==d2):                
                x = state[0]+self.actions[action][0]
                y = state[1]+self.actions[action][1]
                next_state = [x,y]
            reward = self.get_reward(next_state)
            if target_doorway==next_state:
                terminate = True
                #beta[state[0],state[1]] = 1
            else:
                terminate = False
            if (next_state == self.goal_positions[0] or next_state == self.goal_positions[1]):
                done = True
            elif(p < x1 or p > x2 or q < y1 or q > y2):
                # Transitions that take you off the grid will not result in any change and wont result in goal state
                reward = -0.1
                next_state = state
                # print("wall")
                # print(state, action)
                done = False
                terminate = False
            else : 
                # steps leading to stay inside the room
                x = state[0] + self.actions[action][0]
                y = state[1] + self.actions[action][1] 
                next_state = [x,y]
                reward = self.get_reward(next_state)
                # print("will change", self.actions[action][0], self.actions[action][1], state , next_state)
                if (next_state == self.goal_positions[0] or next_state == self.goal_positions[1]):
                    done = True
                else:
                    done = False 
                terminate = False
        # print(next_state)
        return next_state, reward, done, terminate

    def reset(self):
        # select a random start state
        pos = self.get_start_position()
        return pos

# if __name__=='__main__':
#     obj = Fourrooms()
#     obj.get_start_position()