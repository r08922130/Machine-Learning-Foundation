import numpy as np
import matplotlib.pyplot as plt
import  multiprocessing as mp


def sign(x):
    return np.array([1 if data >= 0 else -1 for data in x])

#positive and negative rays 
def perceptron_1D(s, theta, x):
    return s * sign(x-theta)

def error_in(x,y,s,theta,num=2000):
    h = perceptron_1D(s,theta, x)
    #print("==========H==========")
    #print(h==y)
    return h,num - sum(h==y)

# problem 6 (a) generate x ba a uniform distribution
def generate_x(data_num=2000):
    return np.random.uniform(-1,1,data_num)

# problem 6 (b) flips 20% of results
def f(x):
    """length = x.shape
    noise = np.ones(length)
    noise[:int(length[0]/5)] *= -1    
    np.random.shuffle(noise)"""
    noise = np.random.rand(x.shape[0])
    #print("========noise=========")
    
    #print(noise)
    noise = noise<0.2
    noise = noise * -2 +1
    #print("========noise=========")
    
    #print(noise)
    return noise * sign(x)
def error_out(s,theta):
    return 0.5 + 0.3* s *( abs(theta) -1 )

def train(time,num):
    np.random.seed(time)
    x = generate_x(data_num=num)
    x = np.sort(x)
    
    s = [1,-1]
    best_theta = x[0]
    best_s = s[0]
    min_error = num
    #print("==========X==========")
    #print(x)
    y = f(x)
    #print("==========Y==========")
    #print(y)
    #print(time)
    for j, s_t in enumerate(s):
        h, error = error_in(x,y,s_t,x[0],num=num)
        #print("==========H==========")
        #print(h)
        if error < min_error:
            min_error = error
            best_theta = x[0]
            best_s = s_t
        #print(error)
        for i in range(len(h)):
            h[i] = -1 * h[i]
            error -= (1 if h[i]==y[i] else -1)
            #print(i,error)
            if error < min_error:
                min_error = error
                best_theta = x[i]
                best_s = s_t
                #print(min_error)
    e_in = min_error / num
    e_out = error_out(best_s, best_theta)
    #print(e_in)
    return e_in ,e_in - e_out, e_out
##### problem 7 and 8 ######


if __name__ == '__main__':
    
    data_num = [20,2000]
    for k,num in enumerate(data_num):
        p = mp.Pool(processes = 10)
        pool_result = []
        #print(num)
        err_hist = []
        e_in_arr = []
        e_out_arr = []
        for i in range(1000):
            msg = i
            #add event 
            r = p.apply_async(train,(i,num,)) 
            pool_result.append(r)
        #print(len(pool_result))
        for d_err in pool_result:
            #print(d_err.get())
            r = d_err.get()
            e_in_arr += [r[0]]
            err_hist += [r[1]]
            e_out_arr += [r[2]]
        p.close()
        p.join()
        print("e_in:{}".format(np.mean(e_in_arr)))
        print("e_out:{}".format(np.mean(e_out_arr)))
        plt.figure()
        plt.xlabel("difference between E_in and E_out")
        plt.ylabel("times")
        plt.hist(err_hist)

        plt.savefig("problem_{}.jpg".format(k+7))

