import numpy as np
import gym
import random
import time

env = gym.make("FrozenLake-v0")

actionSize = env.action_space.n
stateSize = env.observation_space.n
print("action size:", actionSize)
print("state size:", stateSize)

qTable = np.zeros((stateSize, actionSize))
print("Q Start:", qTable)

# Hyperparameters
totalEpisodes = 60000        # Total episodes ~ total training runs
learningRate = 0.20           # Learning rate
maxSteps = 99                # Max steps per episode ~ max time steps
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
maxEpsilon = 1.0             # Exploration probability at start
minEpsilon = 0.01            # Minimum exploration probability 
decayRate = 0.005             # Exponential decay rate for exploration prob

# List of rewards
rewards = []

start = time.time()
# Training loop
for episode in range(totalEpisodes):
    #reset the env
    state = env.reset()
    step = 0
    done = False #flag to keep track of agent state
    totalRewards = 0 #each training round starts with no reward
    
    
    for step in range(maxSteps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number between 0 and 1
        tradeoff = random.uniform(0, 1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if tradeoff > epsilon:
            action = np.argmax(qTable[state,:])
           # print('action', action)

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
            #print('action', action)
            
        # Take the action (a) and observe the outcome state(s') and reward (r)
        newState, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qTable[state, action] = qTable[state, action] + learningRate * (reward + gamma * np.max(qTable[newState, :]) - qTable[state, action])
        
        totalRewards += reward
        
        # Our new state is state
        state = newState
        
        # If done (if we're dead) : finish episode
        if done == True: 
            break
            
    # Reduce epsilon (because we need less and less exploration)
    epsilon = minEpsilon + (maxEpsilon - minEpsilon)*np.exp(-decayRate*episode) # linear decay
    rewards.append(totalRewards)
    
end = time.time()
print("Training Time:", end - start)
print ("Score over time: " +  str(sum(rewards)/totalEpisodes))
print("Q Final:", qTable)

env.reset()

for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(maxSteps):
        
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qTable[state,:])
        
        newState, reward, done, info = env.step(action)
        
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into a hole)
            #env.render()
            result = 'Win' if newState == 15 else "Loss"
            print("status", result)
            
            # We print the number of step it took.
            print("Number of steps", step)
            break
        state = newState
env.close()