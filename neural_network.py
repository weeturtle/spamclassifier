import numpy as np
from final.activation import TanhActivation, SigmoidActivation
import matplotlib.pyplot as plt

training_spam = np.loadtxt(open("final/data/training_spam.csv"), delimiter=",")
testing_spam = np.loadtxt(open("final/data/testing_spam.csv"), delimiter=",")

total_spam = np.concatenate((training_spam, testing_spam), axis=0)

X = total_spam[:,1:]
Y = total_spam[:,0]


class BinaryNeuralNetwork:
  weights: dict[str, np.ndarray]
  biases: dict[str, np.ndarray]
  layers: list[int]
  update_rate_param: int = 10

  def __init__(self, layers: list[int]) -> None:
    """
    Initializes the neural network with the number of neurons in each layer.
    """
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

    :param X: The input data as a numpy array of shape (n_features, n_samples)

    :return 
      A: The final output layer as a numpy array of shape (1, n_samples)
      caches: A list of tuples containing all of the values used, before and after functions have 
              been applied, in each layer of the network
    """

    A = X
    caches = []

    # Assuming self.layers[-1] is the output layer
    for i in range(1, len(self.layers) - 1):  
        A_Prev = A
        Z = np.dot(self.weights[f"W{i}"], A_Prev) + self.biases[f"b{i}"]

        # Use tanh activation function for all layers except the output layer
        A, activation_cache = TanhActivation.activation_function(Z)
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

    :param A: The predicted output as a numpy array of shape (1, n_samples)
    :param Y: The true output as a numpy array of shape (1, n_samples)

    :return
      cost: The binary cross-entropy loss between the predicted and true outputs
    """
    N = Y.shape[1]
    cost = -1/N * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return cost

  def backwards_propagation_on_layer(self, dL_dA, cache, activation='tanh') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs backward propagation for a single layer.
    
    :param dL_dA: Gradient of the loss with respect to the activation of the current layer.
    :param cache: Tuple containing (linear_hypothesis_cache, activation_cache) from the current layer's forward propagation.
    :param activation: The activation function used in the current layer ('tanh' or 'sigmoid').
    :return: Gradients with respect to the previous layer's activation, current layer's weights, and biases.
    """
    linear_hypothesis_cache, activation_cache = cache
    
    # Choose the derivative function based on the activation parameter
    if activation == 'tanh':
        dA_dZ = TanhActivation.derivative_function(activation_cache)
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

  def backwards_propagation(self, AL, Y, caches) -> dict[str, np.ndarray]:
    """
    The backwards propagation function for the neural network. Uses the final output layer AL and
    the caches from the forward propagation to calculate the gradient descent values for the weight
    and biases of each layer. Just uses batch gradient descent for now.

    :param AL: The final output layer as a numpy array of shape (1, n_samples)
    :param Y: The true output as a numpy array of shape (1, n_samples)
    :param caches: A list of tuples containing all of the values used, before and after functions have 
                  been applied, in each layer of the network

    :return
      grads: A dictionary containing the gradient descent values for the weights and biases of each layer
    """
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dL_dA = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    cur_cache = caches[L-1]
    grads[f"dL_dA{L-1}"], grads[f"dL_dW{L}"], grads[f"dL_db{L}"] = self.backwards_propagation_on_layer(dL_dA, cur_cache, activation='sigmoid')

    for i in range(L-1)[::-1]:
      cur_cache = caches[i]
      dL_dA_prev, dL_dW, dL_db = self.backwards_propagation_on_layer(grads[f"dL_dA{i+1}"], cur_cache, activation='tanh')

      grads[f"dL_dA{i}"] = dL_dA_prev
      grads[f"dL_dW{i+1}"] = dL_dW
      grads[f"dL_db{i+1}"] = dL_db

    return grads
  
  def adjust_weights(self, grads, alpha) -> None:
    """
    Adjust the weights of the layers based on the gradient descent values.

    :param grads: A dictionary containing the gradient descent values for the weights and biases of each layer
    :param alpha: The learning rate for the model
    """
    for i in range(1, len(self.layers)):
      self.weights[f"W{i}"] -= alpha * grads[f"dL_dW{i}"]

  def adjust_biases(self, grads, alpha) -> None:
    """
    Adjust the biases of the layers based on the gradient descent values.

    :param grads: A dictionary containing the gradient descent values for the weights and biases of each layer
    :param alpha: The learning rate for the model
    """
    for i in range(1, len(self.layers)):
      self.biases[f"b{i}"] -= alpha * grads[f"dL_db{i}"]

  def train_model(self,
                  epochs: int,
                  X_data: np.ndarray,
                  Y_data: np.ndarray,
                  alpha=0.01
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each epoch, the forwards propagation is performed and the predicted output and cache is used to calculate the 
    cost and hence the gradient descent values for the weights and biases. The weights and biases are then adjusted
    based on the provided alpha.

    :param epochs: The number of epochs to train the model for
    :param X: The input data as a numpy array of shape (n_features, n_samples)
    :param Y: The true output as a numpy array of shape (1, n_samples)

    :param alpha: The learning rate for the model

    :return
      cost_cache: The cost of the model for each epoch
      accuracy_cache: The accuracy of the model for each epoch
    """

    update_rate = epochs // self.update_rate_param

    cost_cache = np.zeros(update_rate)
    training_accuracy_cache = np.zeros(update_rate)
    testing_accuracy_cache = np.zeros(update_rate)

    # print(f"Shape of X: {X_data.shape}\nShape of Y: {Y_data.shape}")
    X_data = X_data.T
    Y_data = Y_data.reshape(1, Y_data.shape[0])

    j = 0
    for i in range(epochs):
      # Shuffle the data and split into training and testing sets

      # shuffle_indices = np.random.permutation(X_data.shape[1])
      # X_data = X_data[:, shuffle_indices]
      # Y_data = Y_data[:, shuffle_indices]

      X_train = X_data[:, :1400]
      Y_train = Y_data[:, :1400]
      X_test = X_data[:, 1400:]
      Y_test = Y_data[:, 1400:]

      Y_hat, caches = self.forward_propagation(X_train)


      cost = self.cost(Y_hat, Y_train)
      grads = self.backwards_propagation(Y_hat, Y_train, caches)

      self.adjust_weights(grads, alpha)
      self.adjust_biases(grads, alpha)

      if (i % self.update_rate_param == 0):
        print(f"Epoch {i}")
        print(f"Cost: {cost}")
        cost_cache[j] = cost
        training_accuracy_cache[j] = np.mean(np.where(Y_hat > 0.5, 1, 0) == Y_train)
        testing_accuracy_cache[j] = np.mean(np.where(self.predict(X_test.T) > 0.5, 1, 0) == Y_test)
        j += 1


    return cost_cache, training_accuracy_cache, testing_accuracy_cache

  def predict(self, X) -> np.ndarray:
    """
    Predicts the output of the neural network based on the input data X.

    :param X: The input data as a numpy array of shape (n_samples, n_features)

    :return
      Y_hat: The predicted output as a numpy array of shape (1, n_samples)
    """
    Y_hat, _ = self.forward_propagation(X.T)
    return np.where(Y_hat > 0.5, 1, 0)
  
  def save_model(self, filename) -> None:
    """
    Stores the calculated weights and biases of a trained model to a file.

    :param filename: The name of the file to store the model in
    """
    np.savez(filename, **self.weights, **self.biases)

  def load_model(self, filename):
    """
    Retrieves the weights and biases of a trained model from a file and applies them in the model.

    :param filename: The name of the file to retrieve the model from
    """
    data = np.load(filename)
    for key in data.keys():
      if key[0] == 'W':
        self.weights[key] = data[key]
      elif key[0] == 'b':
        self.biases[key] = data[key]


  def generate_plot(self, costs: np.ndarray, training_acc: np.ndarray, test_acc: np.ndarray) -> None:
    """
    Generates a plot of the cost and accuracy of the model over the epochs.
    Used to show if the model is overtraining by comparing the training and testing accuracy.

    :param costs: The cost of the model for each epoch
    :param training_acc: The accuracy of the model on the training data for each epoch
    :param test_acc: The accuracy of the model on the testing data for each epoch
    """
    epochs = range(1, len(training_acc) + 1)

    plt.plot(epochs, costs, label='Cost')
    plt.plot(epochs, training_acc, label='Training Accuracy', linestyle='-')
    plt.plot(epochs, test_acc, label='Test Accuracy', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Cost and Accuracy Over Epochs')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    layers = [54, 20, 1]
    classifier = BinaryNeuralNetwork([54, 20, 1])
    # classifier.load_model("final/models/[54, 20, 1]0.979.npz")
  
    classifier.generate_initial_layers()

    # testing_spam = np.loadtxt(open("final/data/testing_spam.csv"), delimiter=",").astype(int)
    # test_data = testing_spam[:, 1:]
    # test_labels = testing_spam[:, 0]

    costs, training_acc, testing_acc = classifier.train_model(10000, X, Y, 0.15)

    classifier.generate_plot(costs, training_acc, testing_acc)

    y_hat = classifier.predict(X[1400:, :])


    accuracy = np.mean(y_hat == Y[1400:])

    if accuracy > 0.95:
      classifier.save_model(f"final/models/{layers}{round(accuracy, 3)}.npz")


    print(f"Accuracy: {accuracy}")

    # testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
    # test_data = testing_spam[:, 1:]
    # test_labels = testing_spam[:, 0]

    # print(test_data.T.shape)
    # predictions = classifier.predict(test_data)



    # accuracy = np.count_nonzero(predictions == test_labels)/test_labels.shape[0]
    # print(f"Accuracy on test data is: {accuracy}")

