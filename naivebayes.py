import numpy as np

training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")
testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",")

print("Shape of the spam training data set:", training_spam.shape)
print(training_spam)

class SpamClassifier:
  log_class_conditional_likelihoods: np.ndarray
  log_class_priors: np.ndarray
  alpha: float

  def __init__(self, alpha=1.0):
      self.alpha = alpha
  
  def train(self):
      self.log_class_priors = self.estimate_log_class_priors(training_spam)
      self.log_class_conditional_likelihoods = self.estimate_log_class_conditional_likelihoods(training_spam, self.alpha)

  def estimate_log_class_priors(self, data: np.ndarray):
      """
      Given a data set with binary response variable (0s and 1s) in the
      left-most column, calculate the logarithm of the empirical class priors,
      that is, the logarithm of the proportions of 0s and 1s:
          log(p(C=0)) and log(p(C=1))

      :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]
                  the first column contains the binary response (coded as 0s and 1s).

      :return log_class_priors: a numpy array of length two

      """

      # Counts the number of ones and zeros in the first column
      zeros = np.count_nonzero(data[:,0] == 0)
      ones = np.count_nonzero(data[:,0] == 1)
      
      data_length = data.shape[0]
      p_zero = zeros / data_length
      p_one = ones / data_length

      log_class_priors = np.array([np.log(p_zero), np.log(p_one)])

      return log_class_priors



  def estimate_log_class_conditional_likelihoods(self, data: np.ndarray, alpha=1.0):
      """
      Given a data set with binary response variable (0s and 1s) in the
      left-most column and binary features (words), calculate the empirical
      class-conditional likelihoods, that is,
      log(P(w_i | c)) for all features w_i and both classes (c in {0, 1}).

      Assume a multinomial feature distribution and use Laplace smoothing
      if alpha > 0.

      :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]

      :return theta:
          a numpy array of shape = [2, n_features]. theta[j, i] corresponds to the
          logarithm of the probability of feature i appearing in a sample belonging 
          to class j.
      """

      arr = data.copy()

      # Splits the array into 2 arrays, one for spam and one for ham
      c0 = arr[arr[:, 0] == 0]
      c1 = arr[arr[:, 0] == 1]

      c0_length = c0.shape[0]
      c1_length = c1.shape[0]

      k = data.shape[1] - 1

      # Create a theta array to store the probabilities
      theta = np.zeros((2, arr.shape[1] - 1))

      for i in range(1, arr.shape[1]):
          # For the ith feature we only care about columns 0 and i
          
          # Count the number of occurences when wi = 1 for each class c

          c0_wi_occurrences = np.count_nonzero(c0[:, i] == 1)
          c1_wi_occurrences = np.count_nonzero(c1[:, i] == 1)

          # Probability with laplace smoothing
          c0_wi_prob = (c0_wi_occurrences + alpha) / (c0_length + alpha*k)
          c1_wi_prob = (c1_wi_occurrences + alpha) / (c1_length + alpha*k)

          log_c0_wi_prob = np.log(c0_wi_prob)
          log_c1_wi_prob = np.log(c1_wi_prob)

          theta[0, i - 1] = log_c0_wi_prob
          theta[1, i - 1] = log_c1_wi_prob

      return theta

  def predict(self, new_data):
      """
      Given a new data set with binary features, predict the corresponding
      response for each instance (row) of the new_data set.

      :param new_data: a two-dimensional numpy-array with shape = [n_test_samples, n_features].

      :return class_predictions: a numpy array containing the class predictions for each row
          of new_data.
      """

      arr = new_data.copy()
      class_predictions = np.zeros(arr.shape[0])

      for i in range(new_data.shape[0]):
          # For each row in the new data set, calculate the probability of the row belonging to class 0 and class 1
          # using the log class priors and log class conditional likelihoods

          # Multiply each feature element in the row in the test data set with the corresponding 
          # element in the log class conditional for the current class



          c_0_posterior_dist_arr = arr[i] * self.log_class_conditional_likelihoods[0]
          c_1_posterior_dist_arr = arr[i] * self.log_class_conditional_likelihoods[1]

          # Sum the elements
          c_0_posterior_dist = np.sum(c_0_posterior_dist_arr)
          c_1_posterior_dist = np.sum(c_1_posterior_dist_arr)

          # Add the log class priors
          c_0_posterior_dist += self.log_class_priors[0]
          c_1_posterior_dist += self.log_class_priors[1]

          if (c_0_posterior_dist > c_1_posterior_dist):
              class_predictions[i] = 0
          else:
              class_predictions[i] = 1

      return class_predictions


def create_classifier():
  classifier = SpamClassifier(670.0)
  classifier.train()

  return classifier

def accuracy(data: np.ndarray, classifier: SpamClassifier):
    true_classes = data[:, 0]
    features = data[:, 1:]
    class_predictions = classifier.predict(features)
    training_set_accuracy = np.mean(np.equal(class_predictions, true_classes))
    return training_set_accuracy


# Calculate the best alpha value 
classifier = create_classifier()

accs = np.zeros((2, 1000))
j = 0
for i in range(0, 10000, 100):
  alpha = i/10
  classifier.alpha = alpha

  classifier.train()

  acc = accuracy(testing_spam, classifier)

  accs[0, j] = alpha
  accs[1, j] = acc
  j += 1

  print(f"Alpha: {alpha}: {acc*100}%")

# Get the alpha with the highest accuracy
max_acc = np.max(accs[1])
max_acc_index = np.argmax(accs[1])
max_acc_alpha = accs[0, max_acc_index]
print(f"Max accuracy: {max_acc*100}% with alpha: {max_acc_alpha}")

