import numpy as np
import gym
import pyglet
import matplotlib.pyplot as plt
from gym import error,spaces, utils
from gym.utils import seeding
from gym.envs.registration import registry, register, make, spec
from gym.envs.classic_control import rendering


WORLD_PUDDLE = [4, 5, 6] 
puddle_rewards = [-1,-2,-3]                                     # Puddle penalties -1, -2, and -3
puddle_dict = {i:j for i,j in zip(WORLD_PUDDLE,puddle_rewards)}
class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    
    def coord2ind(self, coord):
        [row, col] = coord
        assert(row < self.n)
        assert(col < self.n)
        return col * self.n + row
    
    def ind2coord(self,ind):
        assert(ind>=0)
        col = ind // self.n
        row = ind % self.n
        return [row,col]

    def __init__(self, n=12, noise=0.1, terminal_reward=10.0, border_reward = 0.0, start_state=[6,7,11,12],confusion=0.1,rew1 =-1,rew2 = -2,rew3 = -3,penalty1_states = [39,40,41,42,43,44,45,51,57,63,69,75,81,87,91,92,93,99,100,101,102,103],penalty2_states = [52,53,54,55,56,64,68,76,78,79,80,88,89,90],penalty3_states = [65,66,67,77]): #'random'):
        self.UP = 0
        self.RIGHT = 1
        self.DOWN = 2
        self.LEFT = 3
        self.n = n
        self.terminal_state = [133]
        self.noise = noise
        self.terminal_reward = terminal_reward
        self.border_reward = border_reward
        self.n_states = self.n ** 2 
        self.penalty1_states = penalty1_states
        self.penalty2_states = penalty2_states
        self.penalty3_states = penalty3_states
        self.state = np.random.choice(start_state)
        self.rew1 = rew1
        self.rew2 = rew2
        self.rew3 = rew3
        self.p = 0.9
        self.p_ = (1 -self.p)/3
        self.absorbing_state = self.n_states - 1
        self.done = False
        self.start_state = start_state #if not isinstance(start_state, str) else np.random.rand(n**2)
        self.reset()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.n_states) # with absorbing state
        self.seed()
        self.confusion = confusion
        self.viewer = None
        
        
    def get_reward(self,state):
        if (state in self.penalty1_states):
            reward = -1
        elif (state in self.penalty2_states):
            reward = -2
        elif (state in self.penalty3_states ):
            reward = -3
        elif (state in self.terminal_state):
            reward = 10
        else:
            reward = 0 
        return reward
    
    def next_state(self,state,action):
        
        [row, col] = self.ind2coord(self.state)
        if action == self.UP:
            row = max(row -1, 0)
        elif action == self.DOWN:
            row = min(row + 1, self.n - 1)
        elif action == self.RIGHT:
            col = min(col + 1, self.n - 1)
        elif action == self.LEFT:
            col = max(col - 1, 0) 
        new_state = self.coord2ind([row, col])
        return new_state 
        
    def step(self, action):
        assert self.action_space.contains(action)

        if any(self.state == self.terminal_state):
            self.state = self.absorbing_state
            self.done = True
            info = None
            return self.state, self.get_reward(self.state), self.done,info
        
        if np.random.rand() < self.noise:
            action = self.action_space.sample()
            
        new_state = self.next_state (self.state,action)
        reward = self.get_reward(new_state)
        self.state = new_state
        info = None
        return self.state,reward, self.done,info
        
    def reset(self):
        self.done = False
        self.state = np.random.choice(a=self.start_state) # starting states
        return self.state
    
      
    def seed(self):
        pass
     
    def render(self, mode ='human', close=False):
        p = []
        q = []
        r = []
        s = []
        for i in range(len(self.penalty1_states)):
            p.append(self.ind2coord(self.penalty1_states[i]))


        for i in range(len(self.penalty2_states)):
            q.append(self.ind2coord(self.penalty2_states[i]))
    
        for i in range(len(self.penalty3_states)):
            r.append(self.ind2coord(self.penalty3_states[i]))
    
        for i in range(len(self.terminal_state)):
            s.append(self.ind2coord(self.terminal_state[i]))
        
        
        rew = np.zeros((12,12))
        rew[1,11] = 10
        for i in range(12):
            for j in range(12):
                if ([i,j]) in p:
                    rew[i,j] = -1
                elif ([i,j]) in q:
                    rew[i,j] = -2
                elif([i,j]) in r:
                    rew[i,j] = -3
                elif([i,j]) in s:
                    rew[i,j] = 10
                else:
                    rew[i,j] = 0
        
        plt.imshow(rew)
        plt.show()
        
        

