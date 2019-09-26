import numpy as np
import pandas as pd
import math

def getData(ratio):
    data = pd.read_csv("data.csv")
    # data_use=data['midPrice','AskPrice1', 'BidPrice1']
    midPrice=np.array(data['midPrice'])
    askPrice=np.array(data['AskPrice1'])
    bidPrice=np.array(data['BidPrice1'])
    p=int(midPrice.size*ratio)
    return midPrice[:p], askPrice[:p], bidPrice[:p],midPrice[p:], askPrice[p:], bidPrice[p:]

def getState(mid,t,h):
    """
    mid=midPrice,t=current time,h=history data number
    """
    m=list(mid)
    if (t-h+1>=0):
        state=m[t-h+1:t+1]
    else:
        state= (h-t-1)*[m[0]] + m[0:t+1]
    s=np.array(state)
    if(s.std()==0):
        ss=s-s.mean()
    else:
        ss=(s-s.mean())/s.std()
    ss=np.reshape(ss,(1,h))
    return ss
