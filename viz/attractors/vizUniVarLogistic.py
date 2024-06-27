# https://blbadger.github.io/logistic-map.html
import numpy as np 
import matplotlib.pyplot as plt 

def logistic_map(x, y, step_size, addNoiseLevel=0.00002, multNoiseLevel=0.000005):
	'''a function to calculate the next step of the discrete map.  Inputs
	x and y are transformed to x_next, y_next respectively'''
	y_next = y * x * (1 - y)*np.random.normal(1,multNoiseLevel)+np.random.normal(0, addNoiseLevel)
	x_next = x + step_size
	yield x_next, y_next
	
rate_start=2.5
rate_end=4
steps = 3000000
step_size = (rate_end - rate_start) / steps

Y = np.zeros(steps + 1)
X = np.zeros(steps + 1)

X[0], Y[0] = rate_start, 0.5

# map the equation to array step by step using the logistic_map function above
for i in range(steps):
	x_next, y_next = next(logistic_map(X[i], Y[i], step_size=step_size)) # calls the logistic_map function on X[i] as x and Y[i] as y
	X[i+1] = x_next
	Y[i+1] = y_next
	
plt.style.use('dark_background')
plt.figure(figsize=(10, 10))
plt.plot(X, Y, '^', color='white', alpha=0.4, markersize = 0.013)
plt.axis('on')
plt.show()
plt.savefig('logistic_map.png')