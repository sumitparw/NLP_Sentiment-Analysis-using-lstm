import nltk
import random
import pandas as pd
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from build_dataset import return_train_test_data
import testRNN
from sklearn.metrics import f1_score,accuracy_score, precision_score, recall_score
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

output_file_path = "output_2018.txt"

output_fp = open(output_file_path, "w+")

def f1(classifier, testing_set):
    ground_truth = [r[1] for r in testing_set]
    predictions = {}
    f1_scores = {}

    predictions = [classifier.classify(r[0]) for r in testing_set]
    prec = precision_score(ground_truth, predictions, pos_label="pos")
    print("prec: "+str(prec))
    output_fp.write("\nPrecision: " + str(prec))
    rec = recall_score(ground_truth, predictions, pos_label="pos")
    output_fp.write("\nRecall: " + str(rec))
    print("rec: "+str(rec))
    return f1_score(ground_truth, predictions, pos_label="pos")


training_set, testing_set = return_train_test_data('Dataset/2018.csv')
classifier = nltk.NaiveBayesClassifier.train(training_set)

accuracy = (nltk.classify.accuracy(classifier, testing_set)) * 100
print("Naive Bayes Classifier accuracy percent:", accuracy)
output_fp.write("NBC:\n")
output_fp.write("\nAccuracy: " + str(accuracy))
f1_sc = f1(classifier, testing_set)
print(f1_sc)
output_fp.write("\nF1 Score: " + str(f1_sc))


BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(training_set)
accuracy = (nltk.classify.accuracy(BNB_clf, testing_set)) * 100
print("Bernoulli Classifier accuracy percent:", accuracy)
output_fp.write("\nBC:\n")
output_fp.write("\nAccuracy: " + str(accuracy))
f1_sc = f1(BNB_clf, testing_set)
print(f1_sc)
output_fp.write("\nF1 Score: " + str(f1_sc))

LogReg_clf = SklearnClassifier(LogisticRegression())
LogReg_clf.train(training_set)
accuracy = (nltk.classify.accuracy(LogReg_clf, testing_set)) * 100
print("Logistic Regression Classifier accuracy percent:", )
output_fp.write("\nLR:\n")
output_fp.write("\nAccuracy: " + str(accuracy))
f1_sc = f1(LogReg_clf, testing_set)
print(f1_sc)
output_fp.write("\nF1 Score: " + str(f1_sc))


SGD_clf = SklearnClassifier(SGDClassifier())
SGD_clf.train(training_set)
accuracy = (nltk.classify.accuracy(SGD_clf, testing_set)) * 100
print("SGD Classifier accuracy percent:", accuracy)
output_fp.write("\nSGD:\n")
output_fp.write("\nAccuracy: " + str(accuracy))
f1_sc = f1(SGD_clf, testing_set)
print(f1_sc)
output_fp.write("\nF1 Score: " + str(f1_sc))


SVC_clf = SklearnClassifier(LinearSVC())
SVC_clf.train(training_set)
accuracy = (nltk.classify.accuracy(SVC_clf, testing_set)) * 100
print("SVC Classifier accuracy percent:", accuracy)
output_fp.write("\nSVC:\n")
output_fp.write("\nAccuracy: " + str(accuracy))
f1_sc = f1(SVC_clf, testing_set)
print(f1_sc)
output_fp.write("\nF1 Score: " + str(f1_sc))

# network = testRNN.rnn()
# X_train, Y_train, X_val, Y_val, X_test, Y_test = network.return_train_test_data_rnn('Dataset/2018.csv')
# rnn_model = network.train_model(X_train, Y_train, X_val, Y_val)
# rnn_predictions = network.predict(X_test, rnn_model)
# rnn_accuracy = network.accuracy(rnn_predictions, X_test, Y_test, rnn_model)
# print(rnn_accuracy)

