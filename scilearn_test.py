from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB
import numpy as np

training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")
testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",")


gnb = MultinomialNB()

# Train classifier
print("Training")
gnb.fit(training_spam[:,1:], training_spam[:,0])
print("Training complete")

# Predict class labels
y_pred = gnb.predict(testing_spam[:,1:])

# Display accuracy
print("Accuracy: ", np.mean(y_pred == testing_spam[:,0]))