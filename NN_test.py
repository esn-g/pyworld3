
import numpy as np
import math




# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()

print("a: ",a)
print("b: ",b)

print(x)
y_pred = a + x#b * x

print(y_pred)



'''

for step in kmax/stepsize

    for k in stepsize
        calculate loss
    list of argmins for each step = [autograd of loss of step]

    


    








'''