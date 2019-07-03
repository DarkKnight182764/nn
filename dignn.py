from nn import Nn
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def show(arr,axes=None):
    plt.gray()
    if not axes:
        fig,axes=plt.subplots()                                                              
    arr=np.reshape(arr,(20,20))
    arr=np.transpose(arr)            
    axes.imshow(arr)
    return axes

if __name__ == "__main__":
	res=loadmat("ex4data1.mat")
	X=res["X"];y=res["y"]

	tempy=np.arange(1,11)    
	for i in range(y.shape[0]-1):
		tempy=np.vstack((tempy,np.arange(1,11)))    
	tempy=(tempy==y)+0    
	y=tempy

	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)               
	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	nn=Nn()    
	nn.fit(X_train,y_train,maxfun=50,hidden_units=150,lambdaa=100)     
	print("Done")	

	correct=0
	for row,xi in enumerate(X_train):                
		prediction=list(nn.h(xi).flatten())
		res=prediction.index(max(prediction))+1
		# print("pred:",prediction)		
		# print("y:",list(y_test[row,:]))
		# print("p_res:",res)
		# print("y_res:",list(y_test[row,:]).index(1)+1)		
		# input()				
		if res==list(y_train[row,:]).index(1)+1:
			correct+=1
	print ("Train Accuracy:",correct/X_train.shape[0]*100)

	correct=0
	for row,xi in enumerate(X_test):                
		prediction=list(nn.h(xi).flatten())
		res=prediction.index(max(prediction))+1
		# print("pred:",prediction)		
		# print("y:",list(y_test[row,:]))
		# print("p_res:",res)
		# print("y_res:",list(y_test[row,:]).index(1)+1)		
		# input()				
		if res==list(y_test[row,:]).index(1)+1:
			correct+=1
	print ("Test Accuracy:",correct/X_test.shape[0]*100)

	fig,axes=plt.subplots()
	def repeat(frame):
		#nonlocal axes,X_test,logrs,y_test
		#s=int(input())
		s=frame
		show(X_test[s,:400],axes=axes)     
		prediction=list(nn.h(X_test[frame,:]).flatten())
		res=prediction.index(max(prediction))+1
		print("\n\n\nPredicted:",res)
		print("Correct:",list(y_test[frame,:]).index(1)+1,"\n\n\n")        
		print("pred:",max(prediction))				
		#print(res)
	
	ani=FuncAnimation(fig,repeat,interval=2500)    
	plt.show()
   