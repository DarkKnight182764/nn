import numpy as np
from scipy.io import loadmat
if __name__ == "__main__":
    # t=loadmat("ex4data1.mat")
    # print(t["X"].shape,t["y"].shape)
    # t=loadmat("ex4weights.mat")
    # theta1=t["Theta1"]
    # theta2=t["Theta2"]    
    # print(theta1.shape,theta2.shape)
    x=np.arange(10)
    x=np.reshape(x,(2,5))
    for i,r in enumerate(x):
        for c,v in enumerate(r):
            print(c,v)