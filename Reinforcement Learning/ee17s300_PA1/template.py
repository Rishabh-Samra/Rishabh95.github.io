import numpy as np
import matplotlib.pyplot as plt
from ads import UserAdvert


ACTION_SIZE = 3
STATE_SIZE = 4
TRAIN_STEPS = 10000  # Change this if needed
LOG_INTERVAL = 10

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def learnBandit():
    env = UserAdvert()
    rew_vec = []
    W = np.zeros([4,3])
    for train_step in range(TRAIN_STEPS):
        state = env.getState()
        stateVec = state["stateVec"]
        stateId = state["stateId"]

        # ---- UPDATE code below ------j
        # Sample from policy = softmax(stateVec X W) [W learnable params]
        a = np.transpose(W)*stateVec
        print(a.shape)
        print(W.shape)
        print(stateVec.shape)
        policy = softmax(np.matmul(np.transpose(W)*stateVec))     #3x1
        print(policy)
        # policy = function(stateVec)
        action = int(np.random.choice(3,1,p = policy))
        reward = env.getReward(stateId, action)
        for i in range(4):
             for j in range(3):
                  if(j ==action):
                       W[i,j] = W[i,j]+ reward*(1-policy)*s[i]
                  else:
                       W[i,j] = W[i,j]
         
        # ----------------------------

        # ---- UPDATE code below ------
        # Update policy using reward
        policy = softmax(stateVec*W)
        # ----------------------------

        if train_step % LOG_INTERVAL == 0:
            print("Testing at: " + str(train_step))
            count = 0
            test = UserAdvert()
            for e in range(450):
                teststate = test.getState()
                testV = teststate["stateVec"]
                testI = teststate["stateId"]
                # ---- UPDATE code below ------
                policy = softmax(np.matmul(np.transpose(W)*testV))
                
                # ----------------------------
                act = int(np.random.choice(3,1,p=policy))
                reward = test.getReward(testI, act)
                count += (reward/450.0)
            rew_vec.append(count)

    # ---- UPDATE code below ------
    # Plot this rew_vec list
    print(rew_vec)
    plt.plot(LOG_INTERVAL,rew_vec)


if __name__ == '__main__':
    learnBandit()
