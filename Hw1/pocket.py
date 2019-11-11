import numpy as np
import matplotlib.pyplot as plt

train_PATH="hw1_7_train.dat"
test_PATH = "hw1_7_test.dat"
train_input = np.loadtxt(train_PATH)
test_input = np.loadtxt(test_PATH)


def sign(n):
    return 1 if n > 0 else -1


def perceptron(train_x, train_y,test_x, test_y, w ,update_time):
    error = 1
    w_best = np.copy(w)
    best_error = test_x.shape[0]
    while(update_time > 0):
        error = 0
        for i,data in enumerate(train_x) :
            pred_y = sign(data.dot(w))
            #print(train_y[j])
            if pred_y != train_y[i]:
                error = error + 1
                #modify = modify + 1
                w = w + train_y[i] * data
                w_best , best_error = pocket(w, w_best, best_error,test_x,test_y)
                update_time = update_time - 1
                if update_time == 0 :
                     return best_error/test_x.shape[0]


    #print("Epoch : {}, Modify Time : {} ,Random Time: {}".format(epoch, modify, time))
    return best_error/test_x.shape[0]
def pocket(w, w_best, best_error, test_x,test_y):
    error = 0
    for i,data in enumerate(test_x):
        pred_y = sign(data.dot(w))
        if pred_y != test_y[i]:
            error = error + 1
    #print(error)
    if error < best_error:
        return w , error
    else:
        return w_best , best_error




error_list = []
test_x, test_y = test_input[:,:4], test_input[:,-1]
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis = 1)
for time in range(1126):
    #print(time)
    input_c = np.copy(train_input)
    np.random.seed(time)
    np.random.shuffle(input_c)
    train_x , train_y = input_c[:,:4], input_c[:,-1]
    
    train_x = np.concatenate((np.ones((train_x.shape[0],1)),train_x), axis = 1)
    #print(train_x)
    w = np.array([0,0,0,0,0])
    error_list = error_list + [perceptron(train_x, train_y,test_x, test_y, w,100)]


average_error = sum(error_list) / 1126
fig = plt.hist(error_list,bins=20)
plt.title("Error Rate V.S Frequency = "+ "{:.2f}".format(average_error))
plt.xlabel("Error Rate")
plt.ylabel("Frequency of the Error Rate")
plt.savefig("Hw1_pocket.png")
