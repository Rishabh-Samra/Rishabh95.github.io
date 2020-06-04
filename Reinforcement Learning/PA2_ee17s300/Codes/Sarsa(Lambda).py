


# In[1]:


import gym
import gym_foo
import numpy as np
import time
import os
import matplotlib.pyplot as plt


# In[2]:


episode = 6000


# In[9]:


#Lambda = 0
env = gym.make('FooEnv-v0')
env.reset()
env.seed()
start_time = time.time()
episode = 6000                    
plot_dir = '.\PlotSarsa'
lambd = 0
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    
def get_policy(Q,eps,n_a):
    policy = np.ones(n_a)* eps/n_a
    qmax_index = np.argmax(Q)
    policy[qmax_index]+=1-eps
    return policy

grid_s = 12    
gamma = 0.9
alpha = 0.1
n_a = 4
n_states = grid_s*grid_s
Q = np.random.normal(size = [n_states , n_a])


eps = 0.1
max_steps = 8000
episode_length = np.zeros(episode)
rewepi0 = []
re = []
epi0 = []
e = []
steps = []
for i in range(episode):                                     #episode starts
    state = env.reset()
    E = np.zeros([n_states , n_a])
    #policy = get_policy(Q[state],eps,n_a)
    #action = np.random.choice(a= 4, p = policy)
    end_of_episode = False
    t = 0 
    rewcum = 0
    num_steps = 0 
    while not end_of_episode and num_steps<max_steps:                #steps
        t+=1
        policy = get_policy(Q[state],eps,n_a)  
        action = np.random.choice(a = 4, p = policy)                 #take action a
        new_state ,reward , end_of_episode,_= env.step(action)       
        episode_length[i] = episode_length[i] + 1                   
        rewcum = rewcum + reward 
        pol = get_policy(Q[new_state],eps,n_a)                                 #policy derived from Q
        next_action = np.random.choice(a = 4 ,p = pol)                         # choose A'
        delta = reward + gamma * Q[new_state,next_action] - Q[state,action]
        E[new_state,next_action] = E[new_state,next_action]+1       #accumulating traces
        for s in range(n_states):
            for a in range(n_a):
                Q[s,a] = Q[s,a] + alpha*delta*E[s,a]
                E[s,a] = gamma* lambd* E[s,a]
        state = new_state
        action = next_action
        num_steps = num_steps+1
    re.append(rewcum)
    e.append(episode_length[i])
    steps.append(num_steps)
    if((i+1)%40==0):
        r = np.average(re[i-39:i])
        rewepi0.append(r)
    
        
        ep = np.average(e[i-39:i])
        epi0.append(ep)
        
        
    print("Time{} Episode{} Reward{} Num_steps{}".format(time.time()-start_time,i,re[i],steps[i]))

np.savetxt('Lambda0.txt',rewepi0)
np.savetxt('lmb0.txt',epi0)


# In[15]:


#Lambda = 0.3
env = gym.make('FooEnv-v0')
env.reset()
env.seed()
start_time = time.time()
                    
plot_dir = '.\PlotSarsa'
lambd = 0.3
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    
def get_policy(Q,eps,n_a):
    policy = np.ones(n_a)* eps/n_a
    qmax_index = np.argmax(Q)
    policy[qmax_index]+=1-eps
    return policy

grid_s = 12    
gamma = 0.9
alpha = 0.1
n_a = 4
n_states = grid_s*grid_s
Q = np.random.normal(size = [n_states , n_a])


eps = 0.1
max_steps = 8000
episode_length = np.zeros(episode)
rewepi3 = []
re = []
epi3 = []
e = []
for i in range(episode):                                     #episode starts
    state = env.reset()
    E = np.zeros([n_states , n_a])
    #policy = get_policy(Q[state],eps,n_a)
    #action = np.random.choice(a= 4, p = policy)
    end_of_episode = False
    t = 0 
    rewcum = 0
    num_steps = 0 
    while not end_of_episode and num_steps<max_steps:                #steps
        t+=1
        policy = get_policy(Q[state],eps,n_a)  
        action = np.random.choice(a = 4, p = policy)                           #take action a
        new_state ,reward , end_of_episode,_= env.step(action)       
        episode_length[i] = episode_length[i] + 1                   
        rewcum = rewcum + reward 
        pol = get_policy(Q[new_state],eps,n_a)                                 #policy derived from Q
        next_action = np.random.choice(a = 4 ,p = pol)                         # choose A'
        delta = reward + gamma * Q[new_state,next_action] - Q[state,action]
        E[new_state,next_action] = E[new_state,next_action]+1
        for s in range(n_states):
            for a in range(n_a):
                Q[s,a] = Q[s,a] + alpha*delta*E[s,a]
                E[s,a] = gamma* lambd* E[s,a]
        state = new_state
        action = next_action
        num_steps = num_steps+1
    re.append(rewcum)
    e.append(episode_length[i])
    if((i+1)%40==0):
        r = np.average(re[i-39:i])
        rewepi3.append(r)
        print(rewepi3)
        
        
        ep = np.average(e[i-39:i])
        epi3.append(ep)
        
        
    print("Time{} Episode{} Reward{}".format(time.time()-start_time,i,re[i]))

np.savetxt('Lambda3.txt',rewepi3,delimiter = ',')
np.savetxt('lmb3.txt',epi3,delimiter=',')


# In[3]:



#Lambda = 0.5
env = gym.make('FooEnv-v0')
env.reset()
env.seed()
start_time = time.time()                    
plot_dir = '.\PlotSarsa'
lambd = 0.5
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    
def get_policy(Q,eps,n_a):
    policy = np.ones(n_a)* eps/n_a
    qmax_index = np.argmax(Q)
    policy[qmax_index]+=1-eps
    return policy

grid_s = 12    
gamma = 0.9
alpha = 0.1
n_a = 4
n_states = grid_s*grid_s
Q = np.random.normal(size = [n_states , n_a])


eps = 0.1
max_steps = 8000
episode_length = np.zeros(episode)
rewepi5 = []
re = []
epi5 = []
e = []
for i in range(episode):                                     #episode starts
    state = env.reset()
    E = np.zeros([n_states , n_a])
    #policy = get_policy(Q[state],eps,n_a)
    #action = np.random.choice(a= 4, p = policy)
    end_of_episode = False
    t = 0 
    rewcum = 0
    num_steps = 0 
    while not end_of_episode and num_steps<max_steps:                #steps
        t+=1
        policy = get_policy(Q[state],eps,n_a)  
        action = np.random.choice(a = 4, p = policy)                 #take action a
        new_state ,reward , end_of_episode,_= env.step(action)       
        episode_length[i] = episode_length[i] + 1                   
        rewcum = rewcum + reward 
        pol = get_policy(Q[new_state],eps,n_a)                                 #policy derived from Q
        next_action = np.random.choice(a = 4 ,p = pol)                         # choose A'
        delta = reward + gamma * Q[new_state,next_action] - Q[state,action]
        E[new_state,next_action] = E[new_state,next_action]+1
        for s in range(n_states):
            for a in range(n_a):
                Q[s,a] = Q[s,a] + alpha*delta*E[s,a]
                E[s,a] = gamma* lambd* E[s,a]
        state = new_state
        action = next_action
        num_steps = num_steps+1
    re.append(rewcum)
    e.append(episode_length[i])
    if((i+1)%40==0):
        r = np.average(re[i-39:i])
        rewepi5.append(r)
        
        
        ep = np.average(e[i-39:i])
        epi5.append(ep)

        
    print("Time{} Episode{} Reward{}".format(time.time()-start_time,i,re[i]))
np.savetxt('Lambda5.txt',rewepi5,delimiter = ',')
np.savetxt('lmb5.txt',epi5,delimiter=',')


# In[4]:


#Lambda = 0.9
env = gym.make('FooEnv-v0')
env.reset()
env.seed()
start_time = time.time()
                   
plot_dir = '.\PlotSarsa'
lambd = 0.9
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    
def get_policy(Q,eps,n_a):
    policy = np.ones(n_a)* eps/n_a
    qmax_index = np.argmax(Q)
    policy[qmax_index]+=1-eps
    return policy

grid_s = 12    
gamma = 0.9
alpha = 0.1
n_a = 4
n_states = grid_s*grid_s
Q = np.random.normal(size = [n_states , n_a])


eps = 0.1
max_steps = 8000
episode_length = np.zeros(episode)
rewepi9 = []
re = []
epi9 = []
e = []
for i in range(episode):                                     #episode starts
    state = env.reset()
    E = np.zeros([n_states , n_a])
    #policy = get_policy(Q[state],eps,n_a)
    #action = np.random.choice(a= 4, p = policy)
    end_of_episode = False
    t = 0 
    rewcum = 0
    num_steps = 0 
    while not end_of_episode and num_steps<max_steps:                #steps
        t+=1
        policy = get_policy(Q[state],eps,n_a)  
        action = np.random.choice(a = 4, p = policy)                 #take action a
        new_state ,reward , end_of_episode,_= env.step(action)       
        episode_length[i] = episode_length[i] + 1                   
        rewcum = rewcum + reward 
        pol = get_policy(Q[new_state],eps,n_a)                                 #policy derived from Q
        next_action = np.random.choice(a = 4 ,p = pol)                         # choose A'
        delta = reward + gamma * Q[new_state,next_action] - Q[state,action]
        E[new_state,next_action] = E[new_state,next_action]+1
        for s in range(n_states):
            for a in range(n_a):
                Q[s,a] = Q[s,a] + alpha*delta*E[s,a]
                E[s,a] = gamma* lambd* E[s,a]
        state = new_state
        action = next_action
        num_steps = num_steps+1
    re.append(rewcum)
    e.append(episode_length[i])
    if((i+1)%40==0):
        r = np.average(re[i-39:i])
        rewepi9.append(r)
        
        
        ep = np.average(e[i-39:i])
        epi9.append(ep)
        
        
    print("Time{} Episode{} Reward{}".format(time.time()-start_time,i,re[i]))
np.savetxt('Lambda9.txt',rewepi9)
np.savetxt('lmb9.txt',epi9)


# In[5]:


#Lambda = 0.99
env = gym.make('FooEnv-v0')
env.reset()
env.seed()
start_time = time.time()
                   
plot_dir = '.\PlotSarsa'
lambd = 0.99
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    
def get_policy(Q,eps,n_a):
    policy = np.ones(n_a)* eps/n_a
    qmax_index = np.argmax(Q)
    policy[qmax_index]+=1-eps
    return policy

grid_s = 12    
gamma = 0.9
alpha = 0.1
n_a = 4
n_states = grid_s*grid_s
Q = np.random.normal(size = [n_states , n_a])


eps = 0.1
max_steps = 8000
episode_length = np.zeros(episode)
rewepi99 = []
re = []
epi99 = []
e = []
for i in range(episode):                                     #episode starts
    state = env.reset()
    E = np.zeros([n_states , n_a])
    #policy = get_policy(Q[state],eps,n_a)
    #action = np.random.choice(a= 4, p = policy)
    end_of_episode = False
    t = 0 
    rewcum = 0
    num_steps = 0 
    while not end_of_episode and num_steps<max_steps:                #steps
        t+=1
        policy = get_policy(Q[state],eps,n_a)  
        action = np.random.choice(a = 4, p = policy)                 #take action a
        new_state ,reward , end_of_episode,_= env.step(action)       
        episode_length[i] = episode_length[i] + 1                   
        rewcum = rewcum + reward 
        pol = get_policy(Q[new_state],eps,n_a)                                 #policy derived from Q
        next_action = np.random.choice(a = 4 ,p = pol)                         # choose A'
        delta = reward + gamma * Q[new_state,next_action] - Q[state,action]
        E[new_state,next_action] = E[new_state,next_action]+1
        for s in range(n_states):
            for a in range(n_a):
                Q[s,a] = Q[s,a] + alpha*delta*E[s,a]
                E[s,a] = gamma* lambd* E[s,a]
        state = new_state
        action = next_action
        num_steps = num_steps+1
    re.append(rewcum)
    e.append(episode_length[i])
    if((i+1)%40==0):
        r = np.average(re[i-39:i])
        rewepi99.append(r)
        
        
        ep = np.average(e[i-39:i])
        epi99.append(ep)
        
        
    print("Time{} Episode{} Reward{}".format(time.time()-start_time,i,re[i]))
np.savetxt('Lambda99.txt',rewepi99)
np.savetxt('lmb99.txt',epi99)


# In[7]:


#Lambda = 1
env = gym.make('FooEnv-v0')
env.reset()
env.seed()
start_time = time.time()                    
plot_dir = '.\PlotSarsa'
lambd = 1
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    
def get_policy(Q,eps,n_a):
    policy = np.ones(n_a)* eps/n_a
    qmax_index = np.argmax(Q)
    policy[qmax_index]+=1-eps
    return policy

grid_s = 12    
gamma = 0.9
alpha = 0.1
n_a = 4
n_states = grid_s*grid_s
Q = np.random.normal(size = [n_states , n_a])


eps = 0.1
max_steps = 8000
episode_length = np.zeros(episode)
rewepi1 = []
re = []
epi1 = []
e = []
for i in range(episode):                                     #episode starts
    state = env.reset()
    E = np.zeros([n_states , n_a])
    #policy = get_policy(Q[state],eps,n_a)
    #action = np.random.choice(a= 4, p = policy)
    end_of_episode = False
    t = 0 
    rewcum = 0
    num_steps = 0 
    while not end_of_episode and num_steps<max_steps:                #steps
        t+=1
        policy = get_policy(Q[state],eps,n_a)  
        action = np.random.choice(a = 4, p = policy)                 #take action a
        new_state ,reward , end_of_episode,_= env.step(action)       
        episode_length[i] = episode_length[i] + 1                   
        rewcum = rewcum + reward 
        pol = get_policy(Q[new_state],eps,n_a)                                 #policy derived from Q
        next_action = np.random.choice(a = 4 ,p = pol)                         # choose A'
        delta = reward + gamma * Q[new_state,next_action] - Q[state,action]
        E[new_state,next_action] = E[new_state,next_action]+1
        for s in range(n_states):
            for a in range(n_a):
                Q[s,a] = Q[s,a] + alpha*delta*E[s,a]
                E[s,a] = gamma* lambd* E[s,a]
        state = new_state
        action = next_action
        num_steps = num_steps+1
    re.append(rewcum)
    e.append(episode_length[i])
    if((i+1)%40==0):
        r = np.average(re[i-39:i])
        rewepi1.append(r)
        
        ep = np.average(e[i-39:i])
        epi1.append(ep)
        
        
    print("Time{} Episode{} Reward{}".format(time.time()-start_time,i,re[i]))
np.savetxt('Lambda1.txt',rewepi1)
np.savetxt('lmb1.txt',epi1)


# In[8]:


rewepi0 = np.loadtxt('Lambda0.txt')
epi0 = np.loadtxt('lmb0.txt')
rewepi3 = np.loadtxt('Lambda3.txt')
epi3 = np.loadtxt('lmb3.txt')
rewepi5 = np.loadtxt('Lambda5.txt')
epi5 = np.loadtxt('lmb5.txt')
rewepi9 = np.loadtxt('Lambda9.txt')
epi9 = np.loadtxt('lmb9.txt')
rewepi99 = np.loadtxt('Lambda99.txt')
epi99=np.loadtxt('lmb99.txt')
np.savetxt('Lambda1.txt',rewepi1)
np.savetxt('lmb1.txt',epi1)
rewepi1= np.loadtxt('Lambda1.txt')
epi1 = np.loadtxt('lmb1.txt')


# In[9]:



a = np.linspace(1,6000,150)
plt.plot(a,rewepi0,label = 'LAMBDA = 0')
plt.plot(a,rewepi3,label = 'LAMBDA = 0.3')
plt.plot(a,rewepi5,label = 'LAMBDA = 0.5')
plt.plot(a,rewepi9,label = 'LAMBDA = 0.9')
plt.plot(a,rewepi99,label = 'LAMBDA = 0.99')
plt.plot(a,rewepi1,label = 'LAMBDA = 1')
plt.xlabel('Episode')
plt.ylabel('avg_rew_of_episode')
plt.legend()
plt.savefig(plot_dir +'avg_episode_length.png')
plt.show()
plt.clf()

b = np.linspace(1,6000,150)
plt.plot(b,epi0,label = 'LAMBDA = 0')
plt.plot(b,epi3,label = 'LAMBDA = 0.3')
plt.plot(b,epi5,label = 'LAMBDA = 0.5')
plt.plot(b,epi9,label = 'LAMBDA = 0.9')
plt.plot(b,epi99,label = 'LAMBDA = 0.99')
plt.plot(b,epi1,label = 'LAMBDA = 1')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Average Episode Length')
plt.show()
plt.savefig(plot_dir+'avg length of episode')

_gym_disable_underscore_combat = True 

