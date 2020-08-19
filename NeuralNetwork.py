import numpy as np

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
def sigmoid_derivative(x):
    return x * (1 - x)


def base(inputs):
    inputs = inputs.astype(float) 
    return sigmoid(np.dot(inputs, synaptic_weights))

def train(training_inputs, training_outputs, training_iterations):
    global synaptic_weights
    for iteration in range(training_iterations):
        
        output = think(training_inputs)
        
        error = training_outputs - output
        
        adjustments = np.dot(training_inputs.T, error * sigmoid_derivative(output))

        synaptic_weights += adjustments


print("Random starting synaptic weights: ")
print(synaptic_weights)


training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

  
train(training_inputs, training_outputs, 10000)


print("Synaptic weights after training: ")
print(synaptic_weights)

A = str(input("Input 1: "))
B = str(input("Input 2: "))
C = str(input("Input 3: "))
    
print("New situation: input data = ", A, B, C)
print("Output data: ")
print(think(np.array([A, B, C])))

"""
input 1: 1
input 2: 0
input 3: 0

output:
somthing close to 1

The neural network learns that if there is a 1 in the first colum, that the output should be 1
"""
