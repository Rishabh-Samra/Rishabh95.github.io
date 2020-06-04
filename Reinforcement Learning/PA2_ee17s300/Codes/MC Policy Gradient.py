

# In[ ]:


import gym
import gym_foo
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import pickle


# In[ ]:


def ind2coord(ind):
    assert(ind>=0)
    col = ind // 12
    row = ind % 12
    return [row,col]


# In[ ]:


def softmax(a):
    e = np.exp(a)
    return e/e.sum()


# In[ ]:


def get_policy(theta,state):                                  #function to get policy
    [i,j] = ind2coord(state)
    pr = np.zeros(4)
    pr[0] = theta[0,i-1]+theta[1,j-1]
    pr[1] = theta[2,i-1]+theta[3,j-1]
    pr[2] = theta[4,i-1]+theta[5,j-1]
    pr[3] = theta[6,i-1]+theta[7,j-1]
    prob = softmax(pr)
   #pr = np.argmax(prob)
    return prob


# In[ ]:


env = gym.make('FooEnv-v0')
env.reset()
env.seed()
start_time = time.time()
episode = 5000                    
grid = 12
alpha = 1/64
gamma = 0.98
n_a = 4
state_space = grid*grid
max_steps = 5000
episode_length = np.zeros(episode)
run = 2


# In[ ]:


def get_opt(theta):                                              #function to get optimal policy
    p = []
    for i in range(12):
        for j in range(12):
            pr = np.zeros(4)
            pr[0] = theta[0,i-1]+theta[1,j-1]
            pr[1] = theta[2,i-1]+theta[3,j-1]
            pr[2] = theta[4,i-1]+theta[5,j-1]
            pr[3] = theta[6,i-1]+theta[7,j-1]
            prob = softmax(pr)
            p.append(np.argmax(prob))
    return p


# In[ ]:


UP = 0 
RIGHT = 1
DOWN = 2
LEFT = 3

act = None
rewepi = np.empty([episode,2])
epil = np.empty([episode,2])
rew = np.zeros([run,episode])
epi = np.zeros([run,episode])

for j in range(run):
    re = []
    e = []
    for i in range(episode):
        state = env.reset()
        end_of_episode = False
        ret = 0
        num_steps = 0
        theta = np.random.normal(size = [8,12])

        while not end_of_episode and num_steps < max_steps:                                
            num_steps = num_steps+1          
            prob = get_policy(theta,state)

            action = np.random.choice(a=4,p= prob)                                          
            new_state, reward, end_of_episode,_ = env.step(action)

            ret = ret + np.power(gamma,num_steps)* reward  
            r,s = ind2coord(new_state)
            #update equations
            x = np.exp(theta[action*2,r])/(np.exp(theta[0,r])+np.exp(theta[2,r])+np.exp(theta[4,r])+np.exp(theta[6,r]))    
            y = np.exp(theta[(action*2)+1,s])/(np.exp(theta[1,s])+np.exp(theta[3,s])+np.exp(theta[5,s])+np.exp(theta[7,s]))
            theta[action*2,r] = theta[action*2,r] + alpha * np.power(gamma,num_steps)*ret*(1-x)               #monte carlo update
            theta[action*2+1,s] = theta[action*2+1,s] + alpha *np.power(gamma,num_steps)*ret*(1-y)
            opt= get_opt(theta)
            pol = get_policy(theta,new_state)
            new_act = np.random.choice(a=4,p = pol)
            epi[j,i] = epi[j,i]+1

            state = new_state
            action = new_act
        rew[j,i] = ret
        print(opt)
        print("Time{} Run{} Episode{} Reward{}".format(time.time()-start_time,j,i,ret))
    
rewepi = np.average(rew,axis = 0)
epil = np.average(epi,axis = 0)
    





rewA = np.zeros(1,250)
epiA = np.zeros(1,250)
for i in range(episode):
    if ((i+1)%40==0):
        re = np.average(rewepi[i-39:i])
        rewA[((i+1)/40)-1]=(re)
        ep = np.average(epil[i-39:i])
        epiA[((i+1)/40)-1]=ep




np.savetxt('rewMC.txt',rewepi)
np.savetxt('epilMC.txt',epil)





a = np.linspace(1,8000,200)
plt.plot(a,rewA)
plt.xlabel('Episode')
plt.ylabel('avg_rew_of_episode')
plt.show()
plt.clf()


b = np.linspace(1,8000,200)
plt.plot(b,epiA)
plt.xlabel('Episode')
plt.ylabel('Episode Length')
plt.show()
plt.clf()

