import math
import numpy as np
from random import *
import matplotlib.pyplot as plt
def data_load(path):

    arr = np.loadtxt(path)
    return arr[:,:-1],arr[:,-1]
def sigmoid(s):
    return 1/ (1+np.exp(-s))
def cal_gradient(w,x,y):
    s = np.dot(w,x.T)*y
    gradient_all = -sigmoid(-s).reshape(-1,1)*y.reshape(-1,1)*x
    gradient_avr = np.average(gradient_all,axis=0)
    return gradient_avr
def err_log(w,x,y):
    s = np.dot(w,x.T)*y
    err = -np.log(sigmoid(s))
    error_avr = np.average(err,axis=0)
    return error_avr
def update_W(w,ita,gradient):
    return w- ita*gradient
def err0_1(w,x,y):
    s = np.dot(w,x.T)
    
    scores = sigmoid(s)
    predicts = scores>=0.5
    predicts = predicts *2 -1
    #print(predicts!=y)
    Eout = sum(predicts!=y)
    return (Eout *1.0)/predicts.shape[0]


(x,y) = data_load("hw3_train.dat.txt")
#x0 = 1
x = np.hstack((np.ones(x.shape[0]).reshape(-1,1),x))
step = 2000
lrs = [0.01,0.001]


is_SGD = [False, True]
(test_x,test_y) = data_load("hw3_test.dat.txt")
test_x = np.hstack((np.ones(test_x.shape[0]).reshape(-1,1),test_x))

sign =[ 'k--' ,'k:']
method = ['GD','SGD']
E_in_history = []
E_out_history = []
steps =[]
for lr in lrs:
    for j in range(len(is_SGD)):
        #w0 = 0
        w = np.zeros(x.shape[1])
        
        for i in range(0,step):

            if is_SGD[j]:
                ## Q 20
                x_n = np.array(x[i%x.shape[0]])
                y_n = np.array(y[i%y.shape[0]])
            else :
                ## Q 19
                x_n = x
                y_n = y
            steps += [i+1]
            E_in_history += [err0_1(w,x,y)]
            gradient = cal_gradient(w,x_n,y_n)
            w = update_W(w,lr,gradient)
            E_out_history +=[err0_1(w,test_x,test_y)]
        
        E_out = err0_1(w,test_x,test_y)
        print("=====E_out ( 0/1 err )=====")
        print(E_out)

plt.figure("lr=0.01_Question 7")
plots = plt.plot(steps[:step],E_in_history[:step],sign[0],steps[:step],E_in_history[step:2*step],sign[1])
plt.legend(plots ,(method[0],method[1]))
plt.savefig("lr=0.01_Question_7.png")
plt.figure("lr=0.01_Question 8")
plots = plt.plot(steps[:step],E_out_history[:step],sign[0],steps[:step],E_out_history[step:2*step],sign[1])
plt.legend(plots ,(method[0],method[1]))
plt.savefig("lr=0.01_Question_8.png")
plt.figure("lr=0.001_Question 7")
plots = plt.plot(steps[:step],E_in_history[2*step:3*step],sign[0],steps[:step],E_in_history[3*step:],sign[1])
plt.legend(plots ,(method[0],method[1]))
plt.savefig("lr=0.001_Question_7.png")
plt.figure("lr=0.001_Question 8")
plots = plt.plot(steps[:step],E_out_history[2*step:3*step],sign[0],steps[:step],E_out_history[3*step:],sign[1])
plt.legend(plots ,(method[0],method[1]))
plt.savefig("lr=0.001_Question_8.png")

plt.show()






