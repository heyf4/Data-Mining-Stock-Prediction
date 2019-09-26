from agent import MyAgent
from utils import getData, getState
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt

# position=[5*initial price]
# maxlen of postion is 10

window=60 # history for state
ratio=0.002 # ratio of test and train
batch_size=32
maxEpisode=100

agent=MyAgent(window)
midTrain,askTrain,bidTrain,midTest,askTest,bidTest=getData(ratio)
t_len=len(midTrain)-1

profit_plt=[]

for e in range(maxEpisode):
    print("Episode "+str(e+1)+"/"+str(maxEpisode))
    agent.position=[midTrain[0],midTrain[0],midTrain[0],midTrain[0],midTrain[0]]
    state=getState(midTrain, 0, window)
    # print(state)
    profit=0
    avgPrice=midTrain[0]
    for t in range(t_len):
        action=agent.act(state)
        next_state=getState(midTrain,t+1,window)
        # default reward is zero
        reward=0
        # Action: Buy(2),Sell(1),Hold(0)
        if action==1 and len(agent.position)>0:
                pop_price=agent.position.pop(0)
                real_price=avgPrice
                # reward=max(bidTrain[t]-real_price,0)
                reward=bidTrain[t]-real_price
                profit+=(bidTrain[t]-real_price)
                print("Time:%10d Type:%6s Position:%4d Price:%8.1f Profit:%6.1f Total profit:%16.1f"%(t,"SELL", len(agent.position)-5,bidTrain[t],bidTrain[t]-real_price,profit))
        elif action==2 and len(agent.position)<10:
                reward=avgPrice-askTrain[t]
                totalValue=avgPrice*len(agent.position)
                totalValue+=askTrain[t]
                agent.position.append(askTrain[t])
                avgPrice=totalValue/len(agent.position)
                print("Time:%10d Type:%6s Position:%4d Price:%8.1f"%(t,"BUY", len(agent.position)-5,askTrain[t]))
        # plot
        if(e==(maxEpisode-1)):
                profit_plt.append(profit)

        if t>=(t_len-1):
                done=True
        else:
                done=False
        agent.Memory.append((state,action,reward,next_state,done))
        
        state=next_state

        if done:
                print("Finish:"+str(profit))
        
        if batch_size<len(agent.Memory):
                agent.replay(batch_size)
        
    if (e%30)==0:
        agent.model.save("model_past/episode"+str(e))
                
e_plt=range(len(profit_plt))
plt.plot(e_plt,profit_plt)
plt.title('line chart for last episode')
plt.xlabel('time')
plt.ylabel('profit')
plt.show()




