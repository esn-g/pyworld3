
import numpy as np
import math

import matplotlib as plt

from init_transition_matrix import A_state_transition_matrix

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




def test_normalization(matrix, orig_matrix):  #https://en.wikipedia.org/wiki/Normalization_(statistics)
    for i,var in enumerate(matrix.T):
        scope=max(orig_matrix[:,i])-min(orig_matrix[:,i])
        diff=var-min(orig_matrix[:,i])
        matrix[:,i]=diff/scope
    return matrix



#X=World3_run.generate_state_matrix(runs[0])
#normalized_x=test_normalization(X,X)
normalized_A=test_normalization(A_state_transition_matrix,A_state_transition_matrix)

# Create the heat map using Matplotlib
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
plt.imshow(normalized_A, cmap='viridis', interpolation='nearest', aspect="auto")

# Add a color bar for reference
plt.colorbar()

# Customize axis labels and title
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('12x12 Heat Map')

# Show the plot
plt.show()



'''

for step in kmax/stepsize

    for k in stepsize
        calculate loss
    list of argmins for each step = [autograd of loss of step]

    


    








'''