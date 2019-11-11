import numpy as np
import matplotlib.pyplot as plt

PATH="hw1_6_train.dat"
input_data = np.loadtxt(PATH)
#print(w.dot(train_x[0]))

def sign(n):
    return 1 if n > 0 else -1

def perceptron(train_x, train_y, w ,time):
    error = 1
    modify = 0
    epoch = 0
    while error != 0:
        error = 0
        for i,data in enumerate(train_x) :
            pred_y = sign(data.dot(w))
            #print(train_y[j])
            if pred_y != train_y[i]:
                error = error + 1
                modify = modify + 1
                w = w + train_y[i] * data
        epoch = epoch + 1
    #print("Epoch : {}, Modify Time : {} ,Random Time: {}".format(epoch, modify, time))
    return modify

def shuffle_test():
    #histogram = np.zeros((400,))
    histogram = []
    for i in range(1126):
        input_c = np.copy(input_data)
        np.random.seed(i)
        np.random.shuffle(input_c)
        train_x , train_y = input_c[:,:4], input_c[:,-1]
        train_x = np.concatenate((np.ones((train_x.shape[0],1)),train_x), axis = 1)
        #print(train_x)
        w = np.array([0,0,0,0,0])
        histogram = histogram + [perceptron(train_x, train_y, w,i)]
        #index = perceptron(train_x, train_y, w,i)
        #histogram[index] = histogram[index] + 1
    #print(histogram)
    average_modify = sum(histogram) / 1126
    plt.hist(histogram,bins=20)
    plt.title("Numbers of Update V.S Frequency of the Number, Average = "+ "{:.2f}".format(average_modify))
    plt.xlabel("Numbers of Update")
    plt.ylabel("Frequency of the Number")
    plt.savefig("Hw1.png")

shuffle_test()
