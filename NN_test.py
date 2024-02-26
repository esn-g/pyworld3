
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