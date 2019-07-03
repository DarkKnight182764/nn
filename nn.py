import numpy as np
from scipy.io import loadmat
import random
from scipy.optimize import minimize

class Nn:        
    def unpack(self,theta):        
        theta1=theta[0:self.hu*(self.n+1)]
        theta2=theta[self.hu*(self.n+1):]
        theta1=np.reshape(theta1,(self.hu,self.n+1))
        theta2=np.reshape(theta2,(self.K,self.hu+1))
        return (theta1,theta2)
    
    def dg(self,z):
        return self.g(z)*(1-self.g(z))

    def g(self,z):
        return (1/(1+np.e**(-1*z)))

    # xi(n x 1)
    # returns res(K x 1)
    def h(self,xi,theta=None):      
        xi=np.reshape(xi,(xi.shape[0],1))
        if not theta:
            theta1=self.theta1
            theta2=self.theta2
        else:
            theta1,theta2=theta
        a1=np.vstack((1,xi))        
        z2=np.matmul(theta1,a1)
        a2=self.g(z2)
        a2=np.vstack((1,a2))        

        z3=np.matmul(theta2,a2)
        res=self.g(z3)
        return res

    # y(m x K)
    # x(m x n)
    # regularization includes theta for bias
    def j(self,theta,x,y):
        # print("j")
        lambdaa=0
        theta1,theta2=self.unpack(theta)
        K=y.shape[1]
        m=x.shape[0]
        res=0
        for i in range(m):
            h=self.h(x[i,:][:,None],theta=(theta1,theta2)).flatten()
            for index,v in enumerate(h):
                if v<10**-10:
                    # print("small")
                    h[index]=10**-6
                if v>0.99999999:
                    # print("big")
                    h[index]=0.999999
            h=np.reshape(h,(h.shape[0],1))                                          
            # print(h)
            res=res- np.sum(y[i,:][:,None] * np.log(h) + (1-y[i,:][:,None]) * np.log(1-h))
        # regularize:
        # res+=lambdaa/2*(np.sum(theta1**2)+np.sum(theta2**2))
        #         
        print("j return",res/m)
        return res/m

    def tj(self,theta,x,y):
        print("tj")
        lambdaa=self.lambdaa
        theta1,theta2=theta
        K=y.shape[1]
        m=x.shape[0]
        res=0
        for i in range(m):
            h=self.h(x[i,:][:,None],theta=(theta1,theta2)).flatten()
            for index,v in enumerate(h):
                if v<10**-10:
                    # print("small")
                    h[index]=10**-6
                if v>0.99999999:
                    # print("big")
                    h[index]=0.999999
            h=np.reshape(h,(h.shape[0],1))                                          
            # print(h)
            res=res- np.sum(y[i,:][:,None] * np.log(h) + (1-y[i,:][:,None]) * np.log(1-h))
        # regularize:
        res+=lambdaa/2*(np.sum(theta1**2)+np.sum(theta2**2))
        #         
        print("tj return",res/m)
        return res/m

    def delj(self,theta,x,y):
        # print("delj")
        lambdaa=0
        theta1,theta2=self.unpack(theta)
        T1=np.zeros((self.hu,self.n+1))
        T2=np.zeros((self.K,self.hu+1))

        for i,xi in enumerate(x):
            # xi(n x 1)
            xi=np.reshape(xi,(xi.shape[0],1))
            # a1(n+1 x 1)
            a1=np.vstack((1,xi))        
            z2=np.matmul(theta1,a1)
            a2=self.g(z2)
            # a2 (hidden_units+1  x 1)
            a2=np.vstack((1,a2))        
            z3=np.matmul(theta2,a2)
            # h(K x 1)  = a3
            h=self.g(z3)
            # yi(K x 1)
            yi=y[i,:][:,None]            
            del3=h-yi
            # del2 (hidden_units x 1)
            del2=np.matmul(np.transpose(theta2),del3) * self.dg(np.vstack((1,z2)))
            # remove del2[0]
            del2=del2[1:,:]            
            T1=T1+np.matmul(del2,np.transpose(a1))
            T2=T2+np.matmul(del3,np.transpose(a2))

        # regularize
        T1+=lambdaa*theta1;T2+=lambdaa*theta2
        # 
        T1=T1/self.m;T2=T2/self.m
        # print("delj return")
        # self.gradient_check(T1,T2,x,y,theta1,theta2)
        return np.hstack((T1.flatten(),T2.flatten()))


    # y(m x K)
    # x(m x n)
    # theta1 (hidden_units x n+1)
    # theta2 (K x hidden_units+1)
    def fit(self,x,y,lambdaa=0.1,hidden_units=3,maxfun=20):
        n=x.shape[1];K=y.shape[1];m=y.shape[0];self.lambdaa=lambdaa
        self.n=n;self.m=m;self.K=K;self.hu=hidden_units
        print("y:",y.shape)
        print("x:",x.shape)
        self.theta1=np.random.random_sample((hidden_units,x.shape[1]+1))/20
        self.theta2=np.random.random_sample((y.shape[1],hidden_units+1))/20
        res=minimize(self.j,np.hstack((self.theta1.flatten(),self.theta2.flatten())),args=(x,y),jac=self.delj,
        options={'disp': None, 
        'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 
        'maxfun': maxfun},method="L-BFGS-B")        
        print(res)
        print("min j:",res["fun"])
        self.theta1,self.theta2=self.unpack(res["x"])


    def gradient_check(self,T1,T2,x,y,theta1,theta2):
        tt1=theta1.copy();tt2=theta2.copy()  
        for rowi,row in enumerate(T1):
            for coli,val in enumerate(row):
                eps=10**-6
                tt1[rowi,coli]+=eps                
                cg=(self.tj((tt1,tt2),x,y)-self.tj((theta1,theta2),x,y))/eps
                tt1[rowi,coli]-=eps                                     
                if abs(val-cg)>10**-4:
                    print(abs(val-cg))
                    print(rowi,coli) 
                else:
                    print(abs(val-cg))
        for rowi,row in enumerate(T2):
            for coli,val in enumerate(row):
                eps=10**-6
                tt2[rowi,coli]+=eps                
                cg=(self.tj((tt1,tt2),x,y)-self.tj((theta1,theta2),x,y))/eps
                tt2[rowi,coli]-=eps     
                if abs(val-cg)>10**-4:
                    print(abs(val-cg))
                    print(rowi,coli)
                else:
                    print(abs(val-cg))

if __name__ == "__main__":
    t=loadmat("ex4weights.mat")
    theta1=t["Theta1"]
    theta2=t["Theta2"]    
    res=loadmat("ex4data1.mat")
    X=res["X"];y=res["y"]

    tempy=np.arange(1,11)    
    for i in range(y.shape[0]-1):
        tempy=np.vstack((tempy,np.arange(1,11)))    
    tempy=(tempy==y)+0    
    y=tempy
    nn=Nn()    
    nn.fit(X,y)     