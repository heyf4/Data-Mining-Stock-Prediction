import numpy as np
import pandas as pd
import random

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

from collections import deque


# State:
# [stock_owned=(-5,5),stock_price=midPrice,cash=0]

# Action:
# Buy(2),Sell(1),Hold(0)

class MyAgent:
    def __init__(self, state_num,is_Eval=False, load_model_name=""):
        self.state_num=state_num
        self.action_num=3
        self.position=[]

        self.is_Eval=is_Eval

        self.Memory=deque(maxlen=1000)
        self.gamma=0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005

        if is_Eval:
            self.model=load_model(load_model_name)
        else:
            self.model=self.build_model()


    # LSTM
    # def build_model(self):
    #     model=Sequential()
    #     model.add()


    # MLP
    def build_model(self):
        model=Sequential()
        model.add(Dense(64, input_dim=self.state_num, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(self.action_num, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        return model

    def act(self, state):
        """
        The maximum position is 5 
        """
        if not self.is_Eval:
            if np.random.rand()<self.epsilon:
                return random.randrange(self.action_num)
        act_values=self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self,batch_size):
        mini_batch=random.sample(self.Memory, batch_size)

        for state, action, reward, next_state, done in mini_batch:
            # print(next_state.shape)
            # print(next_state)
            if not done:
                target=reward+self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                target=reward
            label = self.model.predict(state)
            label[0][action] = target
            self.model.fit(state, label, epochs=1, verbose=0)
        self.epsilon_update()
        return

    def epsilon_update(self):
        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay
        return

    def remember(self,state,action,reward,next_state,done):
        self.Memory.append((state,action,reward,next_state,done))
        return
    
