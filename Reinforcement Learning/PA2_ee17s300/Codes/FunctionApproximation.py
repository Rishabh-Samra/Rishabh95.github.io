
# coding: utf-8

# In[28]:


import gym
import gym_foo
import numpy as np
import time
import os
import matplotlib.pyplot as plt
env = gym.make('funap-v0')
env.reset()
env.seed()
start_time = time.time()

plot_dir = './PlotSarsa'




# In[30]:


if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
run=1
episode =60
lambd = 0.3    
eps =0.1
gamma = 0.9
alpha = 0.1
n_a = 4
n_states = 1200*120
Q = np.random.normal(size = [40,n_a])         #Since puddle is divided into 60x60 grids which are 40

eps = 0.1
max_steps = 200000
episode_length = np.zeros(episode)



rewepiA = np.zeros(10)
re = np.zeros(episode)
epi = np.zeros(episode)
epiA = np.zeros(10)

terminal= env.termi_state
puddle1 = env.p1_states
puddle2 = env.p2_states
puddle3 = env.p3_states

phi1 = np.empty([1,40])
phi2 = np.empty([1,40])
phi3 = np.empty([1,40])
phi4 = np.empty([1,40])


def get_policy(Q,eps,n_a):
    policy = np.ones(n_a)* eps/n_a
    qmax_index = np.argmax(Q)
    policy[qmax_index]+= 1-eps
    return policy

def statemid(c):                                       #function to calculate the coordinates of mid points of 60*60 subcells of original 
    a = []
    for i in range(30,1201,60):
        for j in range(30,120,60):
            a.append([i,j])
    return a[c][:]
    
def term_distance(start):                                  #function to calculate the distance of given cell to terminal state
    dis = np.zeros([len(env.terminal_state),1])
    [x0,y0] = start
    for i in range(len(env.terminal_state)):
        [x,y] = terminal[i][:]
        dx = abs(x0-x)
        dy = abs(y0-y)
        dis[i] = np.sqrt(dx**2+dy**2)
    dist = min(dis)
    return dist

def pud1_dist(start):             
    dis = np.zeros([len(env.penalty1_states),1])
    [x0,y0] = start
    for i in range(len(env.penalty1_states)):
        [x,y] = puddle1[i][:]
        dx = abs(x0-x)
        dy = abs(y0-y)
        dis[i] = np.sqrt(dx**2+dy**2)
    dist = min(dis)
    return (dist*-1)
    
def pud2_dist(start):
    dis = np.zeros([len(env.penalty2_states),1])
    [x0,y0] = start
    for i in range(len(env.penalty2_states)):
        [x,y] = puddle2[i][:]
        dx = abs(x0-x)
        dy = abs(y0-y)
        dis[i] = np.sqrt(dx**2+dy**2)
    dist = min(dis)
    return (dist*-1) 
    
def pud3_dist(start):
    dis = np.zeros([len(env.penalty3_states),1])
    [x0,y0] = start
    for i in range(len(env.penalty3_states)):
        [x,y] = puddle3[i][:]
        dx = abs(x0-x)
        dy = abs(y0-y)
        dis[i] = np.sqrt(dx**2+dy**2)
    dist = min(dis)
    return (dist*-1)
    
    
def ind2coord(ind):
        
    col = (ind // 1200)
    row = (ind %1200) - 1
        
    return [row,col]
    
    
def dist(state):                                                 #function for calculating which subgrid any state belongs to
    [x0,y0] = ind2coord(state)
    dis = []
    for i in range(30,1201,60):
        for j in range(30,120,60):
            dx = abs(x0-i)
            dy = abs(y0-j)
        dis.append(np.sqrt(dx**2+dy**2))
    a = np.argmin(dis)
    return a


# In[31]:


for i in range(40):
    phi1[0][i] = term_distance(statemid(i))
    phi2[0][i] = pud1_dist(statemid(i))
    phi3[0][i]= pud2_dist(statemid(i))
    phi4[0][i] = pud3_dist(statemid(i))
    
    
w = np.random.rand(40,4)

phi=np.concatenate((phi1,phi2,phi3,phi4),axis = 0)       
                         
q = w*phi.T                   
                             
v = np.sum(q,axis = 1)
v = np.expand_dims(v,axis = 0)
epi = np.zeros([run,episode])
rew = np.zeros([run,episode])


# In[ ]:


for j in range(run):

    for i in range(episode):                                                    #episode starts
        state = env.reset()
        print(state)
        #policy = get_policy(Q[state],eps,n_a)
        #action = np.random.choice(a= 4, p = policy)
        end_of_episode = False
        e = 0 
        rewcum = 0
        num_steps = 0
    

        while not end_of_episode and num_steps<max_steps:  #steps
            s = dist(state)
            acti = act(s)
            policy = get_policy(Q[s],eps,n_a)  
            action = np.random.choice(a = 4, p = policy)
            new_state,reward,end_of_episode,_= env.step(action) 

            st = dist (new_state)                            
            st1 = dist(state)
           # delta = reward + gamma*v[0,st] - v[0,st1] 
            e = gamma*lambd*e + phi[action,st]
            episode_length[i] = episode_length[i] + 1 
            rewcum = rewcum + reward 
            pol = get_policy(Q[st],eps,n_a)
                                            
            next_action = np.random.choice(a = 4 ,p = pol)                         # choose A'
            delta = reward + gamma * Q[st,next_action] - Q[st1,action]
            
            Q[st1,action] = Q[st1,action] + alpha*delta
            epi[j,i] = epi[j,i]+1
            for p in range(4):                                                     #updating weights of Function Approximator
                w[st1,p] = w[st1,p] + 0.3 *delta * phi[p,st1]    
            state = new_state
            action= next_action                                               
            num_steps = num_steps+1
        rew[j,i] = rewcum
        print(episode_length[i])
        epi[j,i] = episode_length[i]
        opt = np.argmax(Q,axis = 1) 
        print(opt)
        print("Time{} Episode{} Reward{}".format(time.time()-start_time,i,rewcum))

rewepiA = np.average(rew,axis = 0)
epiA = np.average(epi,axis = 0)


# In[ ]:


np.savetxt('FAL3R.txt',rewepiA) 
np.savetxt('FAL3E.txt',epiA)


# In[ ]:



for i in range(episode):
    if (i>1& i<58):
        rewepiA[i] = np.average(rewepiA[i-2:i+2])
        epiA[i] = np.average(epiA[i-2:i+2])
    


# In[ ]:


a = np.linspace(1,60,1)
plt.plot(a,rewepiA)
plt.xlabel('Episode')
plt.ylabel('avg_rew_of_episode')
plt.show()
plt.clf()


b = np.linspace(1,60,1)
plt.plot(b,epiA)
plt.xlabel('Episode')
plt.ylabel('Episode Length')
plt.show()
plt.clf()
env.close()
try:
    del env
except ImportError:
    pass

