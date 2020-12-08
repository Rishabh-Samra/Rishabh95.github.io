



import gym
import gym_foo
import numpy as np
import time
import os
import matplotlib.pyplot as plt



# In[7]:


rewepiA =[]
rewepiB = []
rewepiC = []
epiA = []
epiB = []
epiC = []


#Environment A
env = gym.make('FooEnv-v0')
env.reset()
env.seed()
start_time = time.time()
episode = 6000                    
plot_dir = '.\PlotSarsa'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    
def get_policy(Q,eps,n_a):
    policy = np.ones(n_a)* eps/n_a
    qmax_index = np.argmax(Q)
    policy[qmax_index]+=1-eps
    return policy
    
grid_s = 12    
eps =0.1
gamma = 0.9
alpha = 0.1
n_a = 4
n_states = grid_s*grid_s
Q = np.random.normal(size = [n_states , n_a])


eps = 0.1
max_steps = 9000
episode_length = np.zeros(episode)
rewepiA = []
re = []
e = []
epiA = []
for i in range(episode):                                     #episode starts
    state = env.reset()
    
    end_of_episode = False
    t = 0 
    rewcum = 0
    num_steps = 0
    while not end_of_episode and num_steps<max_steps:                #steps
        t+=1
        policy = get_policy(Q[state],eps,n_a)  
        action = np.random.choice(a = 4, p = policy)
        new_state ,reward , end_of_episode,_= env.step(action) 
        episode_length[i] = episode_length[i] + 1    
        rewcum = rewcum + reward 
        pol = get_policy(Q[new_state],eps,n_a)
        next_action = np.random.choice(a = 4 ,p = pol)
        Q[state,action] = Q[state,action] + alpha*(reward+gamma*Q[new_state,next_action]-Q[state,action])
        a = np.argmax(Q)
        state = new_state
        action = next_action
        num_steps = num_steps+1
    re.append(rewcum)
    e.append(episode_length[i])
    opt = np.argmax(Q,axis = 1)
    print(opt)
    if((i+1)%40==0):
        r = np.average(re[i-39:i])
        rewepiA.append(r)
        
        
        ep = np.average(e[i-39:i])
        epiA.append(ep)
        
    
    print("Time{} Episode{} Reward{}".format(time.time()-start_time,i,rewcum))


# In[5]:


a = np.zeros([4 , 7])
print(a[2,3])


# In[5]:


a = np.linspace(1,6000,150)
plt.plot(a,rewepiA)
plt.xlabel('Episode')
plt.ylabel('avg_rew_of_episode')
plt.savefig(plot_dir +'avg_episode_length.png')
plt.show()
plt.clf()

b = np.linspace(1,6000,150)
plt.plot(b,epiA)
plt.xlabel('Episode')
plt.ylabel('Average Episode Length')
plt.show()
plt.savefig(plot_dir+'avg length of episode')
_gym_disable_underscore_combat = True 





# In[18]:


#Environment B
env = gym.make('FooEnvB-v0')
env.reset()
env.seed()
start_time = time.time()
episode = 6000                    
plot_dir = '.\PlotSarsa'

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
max_steps = 9000
episode_length = np.zeros(episode)
rewepiB = []
re = []
e = []
epiB = []
for i in range(episode):                                     #episode starts
    state = env.reset()
    
    end_of_episode = False
    t = 0 
    rewcum = 0
    num_steps = 0
    while not end_of_episode and num_steps<max_steps:                #steps
        t+=1
        policy = get_policy(Q[state],eps,n_a)  
        action = np.random.choice(a = 4, p = policy)
        new_state ,reward , end_of_episode,_= env.step(action) 
        episode_length[i] = episode_length[i] + 1    
        rewcum = rewcum + reward 
        pol = get_policy(Q[new_state],eps,n_a)
        next_action = np.random.choice(a = 4 ,p = pol)
        Q = Q + alpha*(reward+gamma*Q[new_state,next_action]-Q[state,action])
        state = new_state
        action = next_action
        num_steps = num_steps+1
    re.append(rewcum)
    e.append(episode_length[i])
    opt = np.argmax(Q,axis = 1)
    print(opt)
    if((i+1)%40==0):
        r = np.average(re)
        rewepiB.append(r)
        
        
        ep = np.average(e)
        epiB.append(ep)
       
        
        
        
    
    print("Time{} Episode{} Reward{}".format(time.time()-start_time,i,rewcum))


# In[9]:


np.savetxt('sarsaAre.txt',rewepiA)
np.savetxt('sarsaepA.txt',epiA)
np.savetxt('sarsaBre.txt',rewepiB)
np.savetxt('sarsaepB.txt',epiB)


# In[20]:


#Environment C
env = gym.make('FooEnvC-v0')
env.reset()
env.seed()
start_time = time.time()
episode = 6000                    
plot_dir = '.\PlotSarsa'

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
max_steps = 9000
episode_length = np.zeros(episode)
rewepiC = []
re = []
e = []
epiC = []
for i in range(episode):                                     #episode starts
    state = env.reset()
    
    end_of_episode = False
    t = 0 
    rewcum = 0
    num_steps = 0
    while not end_of_episode and num_steps<max_steps:                #steps
        t+=1
        policy = get_policy(Q[state],eps,n_a)  
        
        action = np.random.choice(a = 4, p = policy)
        new_state ,reward , end_of_episode,_= env.step(action) 
        episode_length[i] = episode_length[i] + 1    
        rewcum = rewcum + reward 
        pol = get_policy(Q[new_state],eps,n_a)
        next_action = np.random.choice(a = 4 ,p = pol)
        Q = Q + alpha*(reward+gamma*Q[new_state,next_action]-Q[state,action])
        state = new_state
        action = next_action
        num_steps = num_steps+1
    re.append(rewcum)
    e.append(episode_length[i])
    if((i+1)%40==0):
        r = np.average(re[i-39:i])
        rewepiC.append(r)
        
        
        ep = np.average(e[i-39:i])
        epiC.append(ep)
        
        
    opt = np.argmax(Q,axis = 1)
    print(opt)
        
    
    print("Time{} Episode{} Reward{}".format(time.time()-start_time,i,rewcum))
np.savetxt('sarsaCre.txt',rewepiC)
np.savetxt('sarsaCepA.txt',epiC)


# In[21]:


newepiA =  np.loadtxt('sarsaAre.txt',delimiter = ',')
epiA = np.loadtxt('sarsaepA.txt',delimiter = ',')
rewepiB = np.loadtxt('sarsaBre.txt',delimiter = ',')
epiB = np.loadtxt('sarsaepB.txt',delimiter = ',')
rewepiC = np.loadtxt('sarsaCre.txt',delimiter = ',')
epiC = np.loadtxt('sarsaCepA.txt',delimiter = ',')


# In[23]:


a = np.linspace(1,6000,150)
plt.plot(a,rewepiA,label = 'envA')
plt.plot(a,rewepiB,label = 'envB')
plt.plot(a,rewepiC,label = 'envC')
plt.xlabel('Episode')
plt.ylabel('avg_rew_of_episode')
plt.legend()
plt.savefig(plot_dir +'avg_episode_length.png')
plt.show()
plt.clf()

b = np.linspace(1,6000,150)
plt.plot(b,epiA,label = 'envA')
plt.plot(b,epiB,label = 'envB')
plt.plot(b,epiC,label = 'envC')
plt.legend()

plt.xlabel('Episode')
plt.ylabel('Average Episode Length')
plt.show()
plt.savefig(plot_dir+'avg length of episode')
_gym_disable_underscore_combat = True 

