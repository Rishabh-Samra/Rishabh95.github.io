
# coding: utf-8

# In[1]:


import gym
import gym_foo
import numpy as np
import time
import os
import matplotlib.pyplot as plt


# In[14]:



env = gym.make('FourRooms-v0')
start_time = time.time()
episode = 30                    


# In[3]:





gamma = 0.9
alpha = 0.1


# In[15]:



def selopt(state,opt):
    I1 = np.arange(0,26,1)
    act = []
    
    terminate = False
    if(opt==1):                  #terminal=103
        a1 = ([2,7,12,17,3,8,13,18,4,9,14,19])
        a2 = ([0,5,10,15])
        a3 = ([1,6,11,16,21])
        a4 = ([20])
        a5 = ([23,24,25])
        if (state in a1):
            act = [0,0,0.5 ,0.5]
        elif (state in a2):
            act = [0,0.5,0.5,0]
        elif (state in a3):
            act = [0,0,1,0]
        elif (state in a4):
            act = [0,1,0,0]
        elif (state in a5):
            act = [0,0,0,1]
        
    
        

    elif(opt==2):

        a6 = ([10,11,12,13,14])
        a7 = ([4,9])
        a8 = ([19,24,103])
        a9 = ([0,1,2,3,5,6,7,8])
        a0 = ([15,16,17,18,20,21,22,23])
        
        if (state in a6):
            act = [0,1,0,0]
        elif (state in a7):
            act = [0,0,1,0]

        elif (state in a8):
            act = [1,0,0 ,0]
        elif (state in a9):
            act = [0,0.5,0.5,0]
        elif (state in a0):
            act = [0.5,0.5,0,0]
        elif (state ==14):
            act = [0,1,0,0]
        elif (state==25):
            terminate = True
      
    elif(opt==3):
        a6 = ([41,46,51,56])
        a7 = ([26,31])
        a8 = ([36,37,38,39,40])
        a9 = ([42,43,44,45,47,48,49,50,52,53,54,55])
        a0 = ([27,28,29,30,32,33,34,35])
        if (state in a6):
            act = [1,0 ,0,0]
        elif (state in a7):
            act = [0,0,1,0]
        elif (state in a8):
            act = [0,0,0,1]
        elif (state in a9):
            act = [0.5,0,0,0.5]

        elif (state in a0):
            act = [0,0,0.5,0.5]
            
        

   
    elif(opt==4):
        a1 = ([29,30,34,35,39,40,44,45,49,50])
        a2 = ([26,27,31,32,36,37,41,42,46,47])
        a3 = ([28,33,38,43,48,53])
        a4 = ([51,52,25])
        a5 = ([54,55])
        if (state in a1):
            act = [0,0,0.5 ,0.5]
        elif (state in a2):
            act = [0,0.5,0.5,0]
        elif (state in a3):
            act = [0,0,1,0]
        elif (state in a4):
            act = [0,1,0,0]
        elif (state in a5):
            act = [0,0,0,1]

    elif(opt==5):
        a1 = ([65,66,70,71,75,76])
        a2 = ([62,63,67,68,72,73])
        a3 = ([59,64,69,74])
        a4 = ([57,58,77])
        a5 = ([60,61])
        if (state in a1):
            act = [0.5,0,0,0.5]
        elif (state in a2):
            act = [0.5,0.5,0,0]
        elif (state in a3):
            act = [1,0,0,0]
        elif (state in a4):
            act = [0,1,0,0]
        elif (state in a5):
            act = [0,0,0,1]
    
    elif(opt==6):
                                  #opt6
        a6 = ([72])
        a7 = ([57,62,56])
        a8 = ([67,68,69,70,71])
        a9 = ([73,74,75,76])
        a0 = ([58,59,60,61,63,64,65,66])
        if (state in a6):
            act = [1,0,0,0]
        elif (state in a7):
            act = [0,0,1,0]
        elif (state in a8):
            act = [0,0,0,1]
        elif (state in a9):
            act = [0.5,0,0,0.5]
        elif (state in a0):
            act = [0,0,0.5,0.5]
        
        
    elif(opt==7):
        a1 = ([98,99,100,101])
        a2 = ([78,79,80,81,83,84,85,86,88,89,90,91])
        a3 = ([102])
        a4 = ([93,94,95,96,97])
        a5 = ([82,87,92,103])
        if (state in a1):
            act = [0.5,0.5,0,0]
        elif (state in a2):
            act = [0,0.5,0.5,0]
        elif (state in a3):
            act = [1,0,0,0]
        elif (state in a4):
            act = [0,1,0,0]
        elif (state in a5):
            act = [0,0,1,0]
        
        
    elif (opt==8):
        a6 = ([78])
        a7 = ([57,62])
        a8 = ([80,81,82,77])
        a9 = ([85,86,87,90,91,92,95,96,97,100,101,102])
        a0 = ([83,88,93,98])
        if (state in a6):
            act = [0,1,0,0]
        elif (state in a7):
            act = [0,0,1,0]
        elif (state in a8):
             act = [0,0,0,1]
        elif (state in a9):
            act = [0.5,0,0,0.5]
        elif (state in a0):
            act = [0.5,0.5,0,0]
            
    return act


# In[16]:




def actpolicy(eps,Q):
    policy = np.ones(4)*eps/4
    q_max_index = np.argmax(Q)
    policy[q_max_index]+=(1-eps)
    return policy


# 
# 
# 
# 
# Option Execution
# 

# In[21]:


def exeopt(state,opt,Q1,Q2,Q3,Q4):
    hallway_states = [25,56,77,103]
    
    list1 = [i for i in range(25)]
    list2 = [i for i in range(26,56)]
    list3 = [i for i in range(57,77)]
    list4 = [i for i in range(78,103)]
                               
    eps =0.5
    
    terminate = False
    alpha = 0.1
    gamma = 0.9
    
    if(opt==1):
        
        
        
                                                                         
        pol = selopt(state,opt)
        action = np.random.choice(4,1,p=pol)
        print(pol)
        
        next_state,reward,end_of_episode,_ = env.step(action)
        pol = selopt(next_state,opt)
        print('NS',next_state)
        next_action = np.random.choice(4,1,p=pol)
        Q1[state,0] = Q1[state,0]+ alpha*(reward + gamma*np.amax([Q1[next_state,0],Q1[next_state,1]])-Q1[state,0])
        
        if(next_state in list4):
            if(a<eps):
                opt = 8
                pol = selopt(next_state,opt)
                next_action = np.random.choice(4,1,p=pol)
                Q4[state,1] = Q4[state,1]+ alpha*(reward + gamma*np.amax([Q4[next_state,0],Q4[next_state,1]])-Q4[state,1])
        
            else:
                opt = 7
                pol = selopt(next_state,opt)
                next_action = np.random.choice(4,1,p=pol)
                Q4[state,0] = Q4[state,0]+ alpha*(reward + gamma*np.amax([Q4[next_state,0],Q4[next_state,1]])-Q4[state,0])
    
        state = next_state        
        action = next_action
        if(state==25):
            a = np.random.uniform(0,1)
            if(a<eps):
                Q1,_,_,_,state,end_of_episode=exeopt(state,1,Q1,Q2,Q3,Q4)
            else:
                _,Q2,_,_,state,end_of_episode=exeopt(state,4,Q1,Q2,Q3,Q4)
        
        
    elif(opt==2):
        
                
        pol = selopt(state,opt)
       # print(2,pol)
        action = np.random.choice(4,1,p=pol)
        next_state,reward,end_of_episode,_ = env.step(action)
        pol = selopt(next_state,opt)
        next_action = np.random.choice(4,1,p=pol)
        Q1[state,1] = Q1[state,1]+ alpha*(reward + gamma*np.amax([Q1[next_state,1],Q1[next_state,0]])-Q1[state,1])
       
        if(next_state in list2):
            if(a<eps):
                opt = 4
                pol = selopt(next_state,opt)
                next_action = np.random.choice(4,1,p=pol)
                Q2[state,1] = Q2[state,1]+ alpha*(reward + gamma*np.amax([Q2[next_state,0],Q2[next_state,1]])-Q2[state,1])
        
            else:
                opt = 3
                pol = selopt(next_state,opt)
                next_action = np.random.choice(4,1,p=pol)
                Q2[state,0] = Q2[state,0]+ alpha*(reward + gamma*np.amax([Q2[next_state,0],Q2[next_state,1]])-Q2[state,0])
        
        state = next_state
        action = next_action
        
    elif(opt==3):
        
        
        pol = selopt(state,opt)
        action = np.random.choice(4,1,p=pol)
        #print('action3',action)
        next_state,reward,end_of_episode,_ = env.step(action)
        pol = selopt(next_state,opt)
        #print(pol)
        #print('3',state,next_state)
        next_action = np.random.choice(4,1,p = pol)
        Q2[state-25,0] = Q2[state-25,0]+ alpha*(reward + gamma*np.amax([Q2[next_state-25,0],Q2[next_state-25,1]])-Q2[state-25,0])
        
        if(next_state in list3):
            if(a<eps):
                opt = 2
                pol = selopt(next_state,opt)
                next_action = np.random.choice(4,1,p=pol)
                Q1[state,1] = Q1[state,1]+ alpha*(reward + gamma*np.amax([Q1[next_state,1],Q1[next_state,0]])-Q1[state,1])
  
            else:
                opt = 1
                pol = selopt(next_state,opt)
                next_action = np.random.choice(4,1,p=pol)
                Q1[state,0] = Q1[state,0]+ alpha*(reward + gamma*np.amax([Q1[next_state,0],Q1[next_state,1]])-Q1[state,0])
  
        
        
        state = next_state
        action = next_action
        
    elif(opt==4):
        
            
        pol = selopt(state,opt)
        #print(pol)
        action = np.random.choice(4,1,p =pol)
        #print('action4',action)
        next_state,reward,end_of_episode,_ = env.step(action) 
        if(next_state in list3):
            if(a<eps):
                opt = 5
            else:
                opt = 6

        pol = selopt(next_state,opt)
        
        #print(pol)
        #print('4',state,next_state)
        next_action = np.random.choice(4,1,p= pol)

        Q2[state-26,1] = Q2[state-26,1]+ alpha*(reward + gamma*np.amax([Q2[next_state-26,1],Q2[next_state-26,0]])-Q2[state-26,1])
        state = next_state
        action = next_action
        
        
    elif(opt==5):
        
        pol = selopt(state,opt)
        action = np.random.choice(4,1,p=pol)
        next_state,reward,end_of_episode,_ = env.step(action)
        if(next_state in list2):
            if(a<eps):
                opt = 4
            else:
                opt = 3
                
        pol = selopt(next_state,opt)
        next_action = np.random.choice(4,1,p=pol)
        
        Q3[(state-57),0] = Q3[(state-57),0]+alpha*(reward + gamma*np.amax([Q3[(next_state-57),1],Q3[(next_state-57),0]])-Q3[(state-57),0])
        
        
        state = next_state
        action = next_action
        
    
    elif(opt==6):
        
        
        pol = selopt(state,opt)
        action = np.random.choice(4,1,p =pol)
        next_state,reward,end_of_episode,_ = env.step(action)
        if(next_state in list4):
            if(a<eps):
                opt = 7
            else:
                opt = 8
        

        pol = selopt(next_state,opt)
        
        next_action = np.random.choice(4,1,p=pol)

        Q3[state-57,1] = Q3[state-57,1]+ alpha*(reward + gamma*np.amax([Q3[next_state-57,1],Q3[next_state-57,0]])-Q3[state-57,1])
        state = next_state
        action = next_action
        
    elif(opt==7):
       
        pol = seloptr4(state,opt)
        action = np.random.choice(4,1,p =pol)
        next_state,reward,end_of_episode,_ = env.step(action)
        if(next_state in list3):
            if(a<eps):
                opt = 6
            else:
                opt = 5

        pol = seloptr4(next_state,opt)
        next_action = np.random.choice(4,1,p= pol)

        Q4[state-78,0] = Q4[state-78,0]+ alpha*(reward + gamma*np.amax([Q4[next_state-78,0],Q4[next_state-78,1]])-Q4[state-78,0])
        state = next_state
        action = next_action
        #print('7')
        
    elif(opt==8):
        
        pol = selopt(state,opt)
        action = np.random.choice(4,1,p =pol)
        next_state,reward,end_of_episode,_ = env.step(action)
        if(next_state in list1):
            if(a<eps):
                opt = 1
            else:
                opt = 2

        pol = selopt(next_state,opt)
        next_action = np.random.choice(4,1, p= pol)

        Q4[state-78,1] = Q4[state-78,1]+ alpha*(reward + gamma*np.amax([Q4[next_state-78,0],Q4[next_state-78,1]])-Q4[state-78,1])
        state = next_state
        action = next_action
        
    return Q1,Q2,Q3,Q4,state,action,end_of_episode


# 
# 
# 
# GOAL2

# In[22]:


n_states = 104
steps = 900
eps = 0.3                                 #for selecting option of a room
for i in range(episode):
    Q  = np.zeros([20,4])                    
    Q1 = np.zeros([25,2]) 
    Q2 = np.zeros([30,2])
    Q3 = np.zeros([20,2])
    Q4 = np.zeros([25,2])
    state = env.reset()
    end_of_episode = False
   
    j = 0
    list1 = [i for i in range(25)]
    list2 = [i for i in range(26,56)]
    list3 = [i for i in range(57,77)]
    list4 = [i for i in range(78,103)]
    
    while not end_of_episode and j<steps:
        j=j+1
        
        if (state in list1):
            
            a = np.random.uniform(0,1)
            if(a<eps):
                Q1,_,_,_,state,action,end_of_episode=exeopt(state,2,Q1,Q2,Q3,Q4)
            else:
                Q1,_,_,_,state,action,end_of_episode=exeopt(state,1,Q1,Q2,Q3,Q4)
            print(state)
            
        elif (state in list2):
            
            b= np.random.uniform(0,1)
            if (b<eps):
                _,Q2,_,_,state,action,end_of_episode=exeopt(state,3,Q1,Q2,Q3,Q4)
            else:
                 _,Q2,_,_,state,action,end_of_episode=exeopt(state,4,Q1,Q2,Q3,Q4)
            print(state)
           
        elif(state in list3):                             #room3
            
            c = np.random.uniform(0,1)

            if(c<eps):
                _,_,Q3,_,state,action,end_of_episode = exeopt(state,6,Q1,Q2,Q3,Q4)
            else:
                _,_,Q3,_,state,action,end_of_episode = exeopt(state,5,Q1,Q2,Q3,Q4)       
            
    
        elif(state in list4):                                     #room4
            
            d = np.random.uniform(0,1)
            if (d<eps):
                _,_,_,Q4,state,action,end_of_episode=exeopt(state,8,Q1,Q2,Q3,Q4)
            else:
                _,_,_,Q4,state,action,end_of_episode=exeopt(state,7,Q1,Q2,Q3,Q4)
                
        elif(state==25):
            e = np.random.uniform(0,1)
            if (e<eps):
                Q1,_,_,_,state,action,end_of_episode=exeopt(state,1,Q1,Q2,Q3,Q4)
            else:
                _,Q2,_,_,state,action,end_of_episode=exeopt(state,4,Q1,Q2,Q3,Q4)
                
        elif(state==103):
            f = np.random.uniform(0,1)
            if (f<eps):
                Q1,_,_,_,state,action,end_of_episode=exeopt(state,2,Q1,Q2,Q3,Q4)
            else:
                _,_,_,Q4,state,action,end_of_episode=exeopt(state,7,Q1,Q2,Q3,Q4)
    
        if(state==77 or state==56):    
            pol = actpolicy(0.1,Q[state])
            action = np.random.choice(4,1,p =pol)
            next_state,reward,end_of_episode,_ = env.step(action)
            pol = actpolicy(0.1,Q[next_state])
            next_action = np.random.choice(a = 4, p= pol)
            Q[state,action] = Q[state,action]+ alpha*(reward+ gamma*(Q[next_state,next_action])-Q[state,action])
            action = next_action
            state = new_state 
        print(state)
        if (end_of_episode == True):       
            print("Episode:{},EpiLen:{}".format(i,j))
        
       # print ("Q1:{},Q2:{},Q3:{},Q4:{},Q:{}".format(Q1,Q2,Q3,Q4,Q))


# In[23]:


a = Q1[:,0]
b = Q1[:,1]
c = Q2[:,0]
d = Q2[:,1]
e = Q3[:,0]

f = Q3[:,1]
g = Q4[:,0]
h = Q4[:,1]


# In[9]:


A = np.reshape(Q1[:,0],(5,5))
B = np.reshape(Q1[:,1],(5,5))
C = np.reshape(Q2[:,0],(6,5))
D = np.reshape(Q2[:,1],(6,5))
E = np.reshape(Q3[:,0],(4,5))
F = np.reshape(Q3[:,1],(4,5))
G = np.reshape(Q4[:,0],(5,5))
H = np.reshape(Q4[:,1],(5,5))
I = np.reshape(Q[:,0],(4,5))
J = np.reshape(Q[:,1],(4,5))
K = np.reshape(Q[:,2],(4,5))
L = np.reshape(Q[:,3],(4,5))


# In[10]:


print(B)


# In[11]:


list = [A,B,C,D,E,F,G,H,Q]
plt.imshow(A);
plt.colorbar()
plt.show()


# In[12]:


plt.imshow(B);
plt.colorbar()
plt.show()


# In[13]:


plt.imshow(C);
plt.colorbar()
plt.show()

