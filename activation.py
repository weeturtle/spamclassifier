import numpy as np

class Activation:
  def activation_function(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The activation function, takes the linear hypothesis Z of each neuron in the layer
    and returns the value applied to each individually and the Z value for the cache

    :param Z: The linear hypothesis of each neuron in the layer (Z = W*X + b) as a numpy array of shape (1, n_neurons)

    :return
      np.ndarray: The value applied to each neuron in the layer as a numpy array of shape (1, n_neurons)
      np.ndarray: The linear hypothesis of each neuron in the layer as a numpy array of shape (1, n_neurons)
    """
    pass

  def derivative_function(Z: np.ndarray) -> np.ndarray:
    """
    The derivative of the activation function, takes the linear hypothesis Z of each neuron in the layer
    and returns the derivative of the value applied to each individually

    :param Z: The linear hypothesis of each neuron in the layer (Z = W*X + b) as a numpy array of shape (1, n_neurons)
    """
    pass


class ReLUActivation(Activation):
  def activation_function(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The rectified linear unit activation function, takes the linear hypothesis Z of each neuron in the layer
    and returns the ReLU value applied to each individually

    :param Z: The linear hypothesis of each neuron in the layer (Z = W*X + b) as a numpy array of shape (1, n_neurons)

    :return 
      relu: The ReLU value applied to each neuron in the layer as a numpy array of shape (1, n_neurons)
      Z: The linear hypothesis of each neuron in the layer as a numpy array of shape (1, n_neurons)
    """
    relu = np.maximum(0, Z)
    return relu, Z
  
  def derivative_function(Z: np.ndarray) -> np.ndarray:
    """
    The derivative of the ReLU activation function, takes the linear hypothesis Z of each neuron in the layer
    and returns the derivative of the ReLU value applied to each individually

    :param Z: The linear hypothesis of each neuron in the layer (Z = W*X + b) as a numpy array of shape (1, n_neurons)

    :return 
      np.ndarray: The derivative of the ReLU value applied to each neuron in the layer as a numpy array of shape (1, n_neurons)
    """
    return np.where(Z > 0, 1, 0)

class SigmoidActivation(Activation):
  def activation_function(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The sigmoid activation function, takes the linear hypothesis Z of each neuron in the layer
    and returns the sigmoid value applied to each individually

    :param Z: The linear hypothesis of each neuron in the layer (Z = W*X + b) as a numpy array of shape (1, n_neurons)

    :return 
      sigmoid: The sigmoid value applied to each neuron in the layer as a numpy array of shape (1, n_neurons)
      Z: The linear hypothesis of each neuron in the layer as a numpy array of shape (1, n_neurons)
    """
    sigmoid = 1/(1 + np.exp(-Z))
    return sigmoid, Z

  def derivative_function(Z: np.ndarray) -> np.ndarray:
    """
    The derivative of the sigmoid activation function, takes the linear hypothesis Z of each neuron in the layer
    and returns the derivative of the sigmoid value applied to each individually

    :param Z: The linear hypothesis of each neuron in the layer (Z = W*X + b) as a numpy array of shape (1, n_neurons)

    :return
      np.ndarray: The derivative of the sigmoid value applied to each neuron in the layer as a numpy array of shape (1, n_neurons)
    """
    sigmoid, _ = SigmoidActivation.activation_function(Z)
    return sigmoid * (1 - sigmoid)

