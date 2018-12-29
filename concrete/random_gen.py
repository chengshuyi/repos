import numpy as np 

def random_gen(mu,sigma,num,dis):
    if dis == '正态分布':
        return np.random.normal(mu,sigma,num)