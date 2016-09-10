import os
import pandas as pd
import numpy as np
from sklearn import linear_model,datasets,tree
import matplotlib.pyplot as plt
iris=datasets.load_iris()
X=iris.data[:,:2]
#X=print(x)
Y=iris.target
#print(Y)
X=X[:100]
Y=Y[:100]
print(X)
print(Y)
number_of_samples=len(Y)
random_indices=np.random.permutation(number_of_samples)
num_training_samples=int(number_of_samples*0.7)
X_train=X[random_indices[:(num_training_samples)]]
Y_train=Y[random_indices[:num_training_samples]]
num_validation_samples = int(number_of_samples*0.15)
X_val = X[random_indices[num_training_samples : num_training_samples+num_validation_samples]]
Y_val = Y[random_indices[num_training_samples: num_training_samples+num_validation_samples]]
#Test set
num_test_samples = int(number_of_samples*0.15)
X_test = X[random_indices[-num_test_samples:]]
Y_test = Y[random_indices[-num_test_samples:]]
X_class0 = np.asmatrix([X_train[i] for i in range(len(X_train)) if Y_train[i]==0]) #Picking only the first two classes
Y_class0 = np.zeros((X_class0.shape[0]),dtype=np.int)
X_class1 = np.asmatrix([X_train[i] for i in range(len(X_train)) if Y_train[i]==1])
Y_class1 = np.ones((X_class1.shape[0]),dtype=np.int)

plt.scatter(X_class0[:,0], X_class0[:,1],color='red')
plt.scatter(X_class1[:,0], X_class1[:,1],color='blue')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(['class 0','class 1'])
plt.title('Fig 3: Visualization of training data')
plt.show()
model=linear_model.LogisticRegression(C=1e5)
full_X=np.concatenate((X_class0,X_class1),axis=0)
full_y=np.concatenate((Y_class0,Y_class1),axis=0)
model.fit(full_X,full_y)
h=.02
x_min,x_max=full_X[:,0].min()-.5,full_X[:,0].max()+.5
y_min,y_max=full_X[:,1].min()-.5,full_X[:,1].max()+.5
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
z=model.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,z,cmap=plt.cm.Paired)
plt.scatter(X_class0[:,0],X_class0[:,1],c='red',edgecolors='k',cmap=plt.cm.Paired)
plt.scatter(X_class1[:,0],X_class1[:,1],c='blue',edgecolors='k',cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title("Visualisation fo decision boundary")
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.show()