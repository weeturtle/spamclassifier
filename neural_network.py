import numpy as np
from activation import Activation, ReLUActivation, SigmoidActivation

training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")
testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",")

X_train = training_spam[:,1:]
X_train = X_train.T
y_train = training_spam[:,0]
y_train = y_train.reshape(1, -1)

X_test = testing_spam[:,1:]
X_test = X_test.T
y_test = testing_spam[:,0]
y_test = y_test.reshape(1, -1)

# Combine the training and testing data to create a single dataset
X = np.concatenate((X_train, X_test), axis=1)
Y = np.concatenate((y_train, y_test), axis=1)

# Split the data into training and testing sets 1300 training, 200 testing
X_train = X[:, :1300]
y_train = Y[:, :1300]
X_test = X[:, 200:]
y_test = Y[:, 200:]



class BinaryNeuralNetwork:
  weights: dict[str, np.ndarray]
  biases: dict[str, np.ndarray]
  layers: list[int]

  def __init__(self, layers: list[int]) -> None:
    self.layers = layers
    self.weights = {}
    self.biases = {}

  def generate_initial_layers(self) -> list[np.ndarray]:
    """
    Generates the initial weights and biases for the neural network layers.
    Adds each weight array to the weights dictionary with the key of W{i} where i is the layer number.
    Adds the biases in the same way with key b{i}.
    """
    for i in range(1, len(self.layers)):
      print(f"Layer {i} has {self.layers[i]} neurons and {self.layers[i-1]} inputs")
      self.weights[f"W{i}"] = np.random.rand(self.layers[i], self.layers[i-1])*0.01
      self.biases[f"b{i}"] = np.zeros((self.layers[i], 1))
  

  def forward_propagation(self, X: np.ndarray) -> tuple[
        np.ndarray, list[
      tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray
      ]
    ]
  ]:
    """
    The forward propagation function for the neural network. Takes the input vector X and applies the
    linear hypothesis Z to each neuron in each layer. Applies the activation function to the Z values
    and returns the final output layer.

    :param X: The input data as a numpy array of shape (1, n_features)

    :return 
      A: The final output layer as a numpy array of shape (1, 1)
      caches: A list of tuples containing all of the values used, before and after functions have 
              been applied, in each layer of the network
    """

    A = X
    caches = []

    # Assuming self.layers[-1] is the output layer
    for i in range(1, len(self.layers) - 1):  
        A_Prev = A
        Z = np.dot(self.weights[f"W{i}"], A_Prev) + self.biases[f"b{i}"]

        # Use ReLU activation function for all layers except the output layer
        A, activation_cache = ReLUActivation.activation_function(Z)
        caches.append(((A_Prev, self.weights[f"W{i}"], self.biases[f"b{i}"]), activation_cache))

    # Output layer using sigmoid
    Z = np.dot(self.weights[f"W{len(self.layers) - 1}"], A) + self.biases[f"b{len(self.layers) - 1}"]
    # Use sigmoid activation function only for output layer
    A, activation_cache = SigmoidActivation.activation_function(Z) 
    caches.append(((A, self.weights[f"W{len(self.layers) - 1}"], self.biases[f"b{len(self.layers) - 1}"]), activation_cache))

    return A, caches

  def cost(self, A: np.ndarray, Y: np.ndarray) -> float:
    """
    The cost function for the neural network. Calculates the binary cross-entropy loss between the
    predicted output A and the true output Y.

    :param A: The predicted output as a numpy array of shape (1, 1)
    :param Y: The true output as a numpy array of shape (1, 1)

    :return
      cost: The binary cross-entropy loss between the predicted and true outputs
    """
    N = Y.shape[1]
    cost = -1/N * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return cost

  def backwards_propagation_on_layer(self, dL_dA, cache, activation='relu'):
    """
    Performs backward propagation for a single layer.
    
    :param dL_dA: Gradient of the loss with respect to the activation of the current layer.
    :param cache: Tuple containing (linear_hypothesis_cache, activation_cache) from the current layer's forward propagation.
    :param activation: The activation function used in the current layer ('relu' or 'sigmoid').
    :return: Gradients with respect to the previous layer's activation, current layer's weights, and biases.
    """
    linear_hypothesis_cache, activation_cache = cache
    
    # Choose the derivative function based on the activation parameter
    if activation == 'relu':
        dA_dZ = ReLUActivation.derivative_function(activation_cache)
    elif activation == 'sigmoid':
        dA_dZ = SigmoidActivation.derivative_function(activation_cache)
    else:
        raise ValueError("Unsupported activation function specified.")

    A_prev, W, b = linear_hypothesis_cache
    N = A_prev.shape[1]

    dL_dZ = dL_dA * dA_dZ

    dL_dW = 1/N * np.dot(dL_dZ, A_prev.T)
    dL_db = 1/N * np.sum(dL_dZ, axis=1, keepdims=True)
    dL_dA_prev = np.dot(W.T, dL_dZ)

    return dL_dA_prev, dL_dW, dL_db

  def backwards_propagation(self, AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dL_dA = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    cur_cache = caches[L-1]
    grads[f"dL_dA{L-1}"], grads[f"dL_dW{L}"], grads[f"dL_db{L}"] = self.backwards_propagation_on_layer(dL_dA, cur_cache, activation='sigmoid')

    for i in range(L-1)[::-1]:
      cur_cache = caches[i]
      dL_dA_prev, dL_dW, dL_db = self.backwards_propagation_on_layer(grads[f"dL_dA{i+1}"], cur_cache, activation='relu')

      grads[f"dL_dA{i}"] = dL_dA_prev
      grads[f"dL_dW{i+1}"] = dL_dW
      grads[f"dL_db{i+1}"] = dL_db

    return grads
  
  def adjust_weights(self, grads, alpha):
    for i in range(1, len(self.layers)):
      self.weights[f"W{i}"] -= alpha * grads[f"dL_dW{i}"]

  def adjust_biases(self, grads, alpha):
    for i in range(1, len(self.layers)):
      self.biases[f"b{i}"] -= alpha * grads[f"dL_db{i}"]

  def train_model(self, X, Y, epochs, alpha=0.01):
    cost_cache = []

    for i in range(epochs):
      Y_hat, caches = self.forward_propagation(X)
      cost = self.cost(Y_hat, Y)
      cost_cache.append(cost)
      grads = self.backwards_propagation(Y_hat, Y, caches)

      self.adjust_weights(grads, alpha)
      self.adjust_biases(grads, alpha)

      if (i % 100 == 0):
        print(f"Epoch {i}")
        print(f"Cost: {cost}")

  def predict(self, X):
    Y_hat, _ = self.forward_propagation(X)
    return Y_hat
  
  def save_model(self, filename):
    np.savez(filename, **self.weights, **self.biases)

  def load_model(self, filename):
    data = np.load(filename)
    for key in data.keys():
      if key[0] == 'W':
        self.weights[key] = data[key]
      elif key[0] == 'b':
        self.biases[key] = data[key]

if __name__ == "__main__":
  model = BinaryNeuralNetwork([54, 50, 1])
  model.load_model("spam_model.npz")
  # model.generate_initial_layers()

  print(f"X_train shape: {X_train.shape} should be (54, 1000)")
  print(f"y_train shape: {y_train.shape} should be (1, 1000)")

  # model.train_model(X_train, y_train, 3500, 0.25)

  y_hat = model.predict(X_test)


  y_hat = np.where(y_hat > 0.5, 1, 0)
  accuracy = np.mean(y_hat == y_test)

  if accuracy > 0.95:
    model.save_model("spam_model.npz")

  print(f"Accuracy: {accuracy}")