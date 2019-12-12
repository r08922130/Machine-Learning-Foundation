Problem 6

(a)function generate(n) :  Design a generating n point as input data with normal distribution function

(b)function f(x) : Use sign(x) to create the ground truth of data. 
	For noise, I create a random array with value between 0 and 1 and give 1 if the value smaller than 0.2 else 0. Then use noise * -2 + 1 to get the 20% noise array with values [1 ,-1] 

Problem 7, 8

Step 1. I sorted the data to increasing order, then choose the smallest one as the initial theta. 

Step 2. Change value of theta to next sorted input data, because the prediction of data will  be changed only when the input data equal to theta, the prediction has only one different value from previous result.  The loss only change on the data which is equal to theta, so I use previous loss to get the current loss by computing E_in_cur = E_in_prev - 1 if the prediction become correct else  E_in_cur = E_in_prev + 1

Step 3. Record the min E_in, best theta and s. Then get Error_out by the best theta and s
. Return the E_in - E_out   