import numpy as np

# Seed the random number generator
np.random.seed(1)

# Set synaptic weights to a 3x1 matrix,
# with values from -1 to 1 and mean 0
synaptic_weights = 2 * np.random.random((3, 1)) - 1

def sigmoid(x):
    """
    Takes in weighted sum of the inputs and normalizes
    them through between 0 and 1 through a sigmoid function
    """
    return 1 / (1 + np.exp(-x))
 
def sigmoid_derivative(x):
    """
    The derivative of the sigmoid function used to
    calculate necessary weight adjustments
    """
    return x * (1 - x)


def think(inputs):
    """
    Pass inputs through the neural network to get output
    """
        
    inputs = inputs.astype(float) 
    return sigmoid(np.dot(inputs, synaptic_weights))

def train(training_inputs, training_outputs, training_iterations):
    global synaptic_weights
    for iteration in range(training_iterations):
        # Pass training set through the neural network
        output = think(training_inputs)

        # Calculate the error rate
        error = training_outputs - output

        # Multiply error by input and gradient of the sigmoid function
        # Less confident weights are adjusted more through the nature of the function
        adjustments = np.dot(training_inputs.T, error * sigmoid_derivative(output))

        # Adjust synaptic weights
        synaptic_weights += adjustments

# Initialize the single neuron neural network


print("Random starting synaptic weights: ")
print(synaptic_weights)

# The training set, with 4 examples consisting of 3
# input values and 1 output value
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

    # Train the neural network
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