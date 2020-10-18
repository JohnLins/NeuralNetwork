import numpy as np

np.random.seed(1)

synapticWeights = 2 * np.random.random((3, 1)) - 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
def sigmoidDerivative(x):
    return x * (1 - x)


def base(inputs):
    inputs = inputs.astype(float) 
    return sigmoid(np.dot(inputs, synapticWeights))

def train(trainingInputs, trainingOutputs, trainingIterations):
    global synapticWeights
    for iteration in range(trainingIterations):
        
        output = base(trainingInputs)
        
        error = trainingOutputs - output
        
        adjustments = np.dot(trainingInputs.T, error * sigmoidDerivative(output))

        synapticWeights += adjustments


print("Random starting weights: ")
print(synapticWeights)


trainingInputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

trainingOutputs = np.array([[0,1,1,0]]).T

  
train(trainingInputs, trainingOutputs, 10000)


print("Weights after training: ")
print(synapticWeights)
print("-------------------")

input1 = str(input("Input 1: "))
input2 = str(input("Input 2: "))
input3 = str(input("Input 3: "))
    
print("Input data = ", input1, input2, input3)
print("-------------------")
print("Output data: ")
print(base(np.array([input1, input2, input3])))

"""
input 1: 1
input 2: 0
input 3: 0

Output:
somthing close to 1

The neural network learns that if there is a 1 in the first column, that the output should be 1
"""
