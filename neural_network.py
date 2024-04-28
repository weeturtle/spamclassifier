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

class BinaryNeuralNetwork:
  weights: dict[str, np.ndarray]
  biases: dict[str, np.ndarray]
  layers: list[int]
  activation: Activation

  def __init__(self, layers: list[int], activation: Activation = SigmoidActivation) -> None:
    self.layers = layers
    self.weights = {}
    self.biases = {}
    self.activation = activation

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

    for i in range(1, len(self.layers)):
      A_Prev = A
      # print(f"Multiply {self.weights[f'W{i}']} by {A_Prev}")
      Z = np.dot(self.weights[f"W{i}"], A_Prev) + self.biases[f"b{i}"]
      # print(f"Means {Z}")

      linear_hypothesis_cache = (A, self.weights[f"W{i}"], self.biases[f"b{i}"])
      A, activation_cache = self.activation.activation_function(Z)

      caches.append((linear_hypothesis_cache, activation_cache))

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
  
  def backwards_propagation_on_layer(self, dL_dA, cache):
    linear_hypothesis_cache, activation_cache = cache
    
    dA_dZ = self.activation.derivative_function(activation_cache)

    A_prev, W, b = linear_hypothesis_cache
    N = A_prev.shape[1]

    dL_dZ = dL_dA * dA_dZ

    dL_dW = 1/N * np.dot(dA_dZ, A_prev.T)
    dL_db = 1/N * np.sum(dA_dZ, axis=1, keepdims=True)
    dL_dA_prev = np.dot(W.T, dL_dZ)

    return dL_dA_prev, dL_dW, dL_db
  
  def backwards_propagation(self, AL, Y, caches):
    grads = {}
    L = len(caches)
    N = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dL_dA = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    cur_cache = caches[L-1]
    grads[f"dL_dA{L-1}"], grads[f"dL_dW{L}"], grads[f"dL_db{L}"] = self.backwards_propagation_on_layer(dL_dA, cur_cache)

    for i in range(L-1)[::-1]:
      cur_cache = caches[i]
      dL_dA_prev, dL_dW, dL_db = self.backwards_propagation_on_layer(grads[f"dL_dA{i+1}"], cur_cache)

      grads[f"dL_dA{i}"] = dL_dA_prev
      grads[f"dL_dW{i+1}"] = dL_dW
      grads[f"dL_db{i+1}"] = dL_db

    return grads
  
  def adjust_weights(self, grads, alpha):
    for i in range(1, len(self.layers)):
      print(f"Adjusting weights for layer {i}")
      self.weights[f"W{i}"] -= alpha * grads[f"dL_dW{i}"]

  def adjust_biases(self, grads, alpha):
    for i in range(1, len(self.layers)):
      self.biases[f"b{i}"] -= alpha * grads[f"dL_db{i}"]

  def train_model(self, X, Y, epochs, alpha=0.01):
    cost_cache = []

    for i in range(epochs):
      print(f"Epoch {i}")

      Y_hat, caches = self.forward_propagation(X)
      cost = self.cost(Y_hat, Y)
      cost_cache.append(cost)
      grads = self.backwards_propagation(Y_hat, Y, caches)

      self.adjust_weights(grads, alpha)
      self.adjust_biases(grads, alpha)

      print(f"Cost: {cost}")

  def predict(self, X):
    Y_hat, _ = self.forward_propagation(X)
    return Y_hat

if __name__ == "__main__":
  model = BinaryNeuralNetwork([54, 20, 1])
  model.generate_initial_layers()

  print(f"X_train shape: {X_train.shape} should be (54, 1000)")
  print(f"y_train shape: {y_train.shape} should be (1, 1000)")

  model.train_model(X_train, y_train, 100, 0.05)

  y_hat = model.predict(X_test)
  # print(y_hat)
  # print(y_test)

  y_hat = np.where(y_hat > 0.5, 1, 0)
  accuracy = np.mean(y_hat == y_test)
  print(f"Accuracy: {accuracy}")