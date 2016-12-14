""" Udacity Machine Learning Puzzle #Tensor Flow
Softmax
Note: Your softmax(x) function should return a NumPy array of the same shape as x.

For example, given a list or one-dimensional array (which is interpreted as a column vector representing a single sample), like:

scores = [1.0, 2.0, 3.0]
It should return a one-dimensional array of the same length, i.e. 3 elements:

print softmax(scores)
[ 0.09003057  0.24472847  0.66524096]
Given a 2-dimensional array where each column represents a sample, like:

scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])
It should return a 2-dimensional array of the same shape, (3, 4):

[[ 0.09003057  0.00242826  0.01587624  0.33333333]
 [ 0.24472847  0.01794253  0.11731043  0.33333333]
 [ 0.66524096  0.97962921  0.86681333  0.33333333]]
The probabilities for each sample (column) must sum to 1. Feel free to test your function with these inputs. 
"""

scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x) , axis=0) 
      

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
