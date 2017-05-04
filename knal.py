import numpy as np
import random as ran
from matplotlib import pyplot as p
import sys


#Method for creating a set of all features
def feature_set(file):
    features = set()
    with open(file) as features_file:
        for line in features_file:
            for word in line.strip().split():
                features.add(word)
    return features

#Method for creating a unions for features sets
def union(set_a, set_b):
    return list(set(set_a) | set(set_b))

#Method for getting weights:
def feature_weights(all_features):
    weight_dic = {}
    for (index, value) in enumerate(all_features):
        weight_dic[value] = 0 #setting weights' values to 0
    return weight_dic

#Method for getting vectors from reviews
def vectors(file, label):
    feat_vectors = []
    with open(file) as features_file:
        for line in features_file:
            features = {}
            for word in line.strip().split():
                features[word] = 1 #because each word in the review appears only once
            feat_vectors.append((features,label))
    return feat_vectors

#Perceptron Class    
class Perceptron():
 
    def __init__(self, weights, training_data, testing_data, bias, training_errors,\
    num_of_train_errors, learning_rate, testing_errors, num_of_test_errors,\
    correct_label, total_correct, accuracies, iterations):

        self.weights = weights
        self.training_data = training_data
        self.testing_data = testing_data
        self.bias = bias
        self.num_of_train_errors = num_of_train_errors
        self.training_errors = training_errors
        self.learning_rate = learning_rate
        self.testing_errors = testing_errors
        self.num_of_test_errors = num_of_test_errors
        self.total_correct = total_correct
        self.accuracies = accuracies
        self.correct_label = correct_label
        self.iterations = iterations
        
    def train(self):
        for review in self.training_data:
            features = review[0]
            expected_label = review[1]
            activation = sum(features[feature] * self.weights[feature] for feature in features)
            activation += self.bias
        
            if ((activation * expected_label) <= 0):
                self.num_of_train_errors += 1
                
                for feature in features:
                    self.weights[feature] = self.weights[feature] + self.learning_rate * expected_label * features[feature]
                self.bias += expected_label
        
        training_error = (self.num_of_train_errors / len(self.training_data)) * 100
        self.training_errors.append(training_error)
        
        return (self.weights, self.bias, self.training_errors)
    
    def test(self):
        
        for review in self.testing_data:
            features = review[0]
            expected_label = review[1]
            activation = sum(features[feature] * self.weights[feature] for feature in features)
            activation += self.bias
            
            if (np.sign(activation) == expected_label):
                self.correct_label += 1
            
            else:
                self.num_of_test_errors += 1
            
        
        testing_error = (self.num_of_test_errors / len(self.testing_data)) * 100
        self.testing_errors.append(testing_error)
        accuracy = (self.correct_label / len(self.testing_data)) * 100
        self.accuracies.append(accuracy)
        self.total_correct.append(self.correct_label)
        
        return (self.accuracies, self.total_correct, self.testing_errors)
    
    def error_graph(self):

        p.figure()
        p.plot(range(1, self.iterations + 1), self.training_errors, '--g', label='Training Errors')
        p.plot(range(1, self.iterations + 1), self.testing_errors, '--b', label='Testing Errors')
        p.axis([1, self.iterations, 0, (max(max(self.training_errors), max(self.testing_errors))+5)])
        p.xlabel('Number of Iterations')
        p.ylabel('Error Rate %')
        p.legend(loc='upper right')
    
#Method for the classifier evaluation
def classifier_evaluation(precisions_TP, recalls_TP, false_positive_rates):
    
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    
    for review in testing_data:
        features = review[0]
        expected_label = review[1]

        activation = sum(features[feature] * weights[feature] for feature in features)
        activation = activation + bias
        
        if expected_label == 1:    
            if (np.sign(activation) == expected_label):
                true_positive += 1
            else:
                false_positive += 1
                
        if expected_label == -1:
            if (np.sign(activation) == expected_label):
                true_negative += 1
            else:
                false_negative += 1
            
    precision_TP = (true_positive / (true_positive + false_positive) * 100)
    precisions_TP.append(precision_TP)
    recall_TP = (true_positive / (true_positive + false_negative))
    recalls_TP.append(recall_TP)
    false_positive_rate = (false_positive / (false_positive + true_negative))
    false_positive_rates.append(false_positive_rate)

    return (precisions_TP, recalls_TP, false_positive_rates)

#Method for the classifier evaluation- visualization
def eval_graphs(accuracies, precisions_TP, recalls_TP, false_positive_rates):
    
    p.figure()
    p.subplot(2, 2, 1)
    p.plot(range(1, iterations + 1), accuracies, '--g', label='Accuracy')

    p.xlabel('Number of Iterations')
    p.ylabel('Accuracy %')
    p.tight_layout()
    
    p.subplot(2, 2, 2)
    p.plot(range(1, iterations + 1), precisions_TP, '--r', label='Precision')
    p.xlabel('Number of Iterations')
    p.ylabel('Precision %')
    p.tight_layout()
    
    p.subplot(2, 2, 3)
    p.plot(range(1, iterations + 1), recalls_TP, '--y', label='Recall')
    p.xlabel('Number of Iterations')
    p.ylabel('Recall')
    p.tight_layout()
    
    p.subplot(2, 2, 4)
    p.plot(range(1, iterations + 1), false_positive_rates, '--b', label='False Positive Rate')
    p.xlabel('Number of Iterations')
    p.ylabel('False Positive Rate')
    p.tight_layout()

#Method for printing progress bar
def progress(count, iterations, suffix=''):
    bar_length = 60
    filled_bar_length = int(round(bar_length * count / (iterations)))

    percentage = round(100.0 * count / (iterations), 1)
    progress_bar = '=' * filled_bar_length + '-' * (bar_length - filled_bar_length)

    sys.stdout.write('[%s] %s%s ...%s\r' % (progress_bar, percentage, '%', suffix))
    sys.stdout.flush() 


##################################################################################################################
if __name__ == "__main__":
    #Gettting the features and unions to determine the feature set
    train_pos_feat = feature_set("./data/train.positive")
    train_neg_feat = feature_set("./data/train.negative")
    test_pos_feat = feature_set("./data/test.positive")
    test_neg_feat = feature_set("./data/test.negative")
    
    #Geting unions of the feature sets
    training = union(train_pos_feat, train_neg_feat)
    testing = union(test_pos_feat, test_neg_feat)
    all_features = union(training, testing)
    
    #Setting feature weights to 0
    weights = feature_weights(all_features)
    
    #Getting the feature vectors for testing and training data
    train_pos = vectors("./data/train.positive", 1)
    train_neg = vectors("./data/train.negative", -1)
    test_pos = vectors("./data/test.positive", 1)
    test_neg = vectors("./data/test.negative", -1)
    
    #Generating training and testing data sets
    train_pos.extend(train_neg)
    training_data = train_pos
    test_pos.extend(test_neg)
    testing_data = test_pos
    
    #Perceptron Training Info
    learning_rate = float(input("\nPlease specify the learning rate to train the algorithm, for example 0.1 or 0.01 etc.: "))
    print(learning_rate)
    bias = 0
    num_of_train_errors = 0
    training_errors = [] # percentage error
    
    
    #Perceptron Testing Info
    num_of_test_errors = 0
    correct_label = 0
    testing_errors = [] # percentage error
    accuracies = []
    total_correct = []
    precisions_TP= []
    recalls_TP = []
    false_positive_rates = []
    
    #User Input for Iterations
    iterations = int(input("\nPlease specify the number of iterations to train the algorithm: "))
    print (iterations)
    
    #Calling the Perceptron Class
    for i in range(iterations):
        progress (i+1, iterations, suffix = '')
        x = Perceptron(weights, training_data, testing_data, bias, training_errors,\
        num_of_train_errors, learning_rate, testing_errors, num_of_test_errors,\
        correct_label, total_correct, accuracies, iterations)
        ran.shuffle(training_data)
        x.train()
        ran.shuffle(testing_data)
        x.test()
        evaluation = classifier_evaluation(precisions_TP, recalls_TP, false_positive_rates)


    print ("\n##########     TESTING INFORMATION     ##########")
    print ("** average results from %d training iterations **" % (iterations))
    
    print ("\n<ACCURACY = %.2f%%>" %  (sum(accuracies) / len(accuracies)))
    print ("\n<CLASSIFICATION ERROR = %.2f%%>" % (100 - (sum(accuracies) / len(accuracies))))
    print ("\n<PRECISION = %.2f%%>" % (sum(precisions_TP) / len(precisions_TP)))
    print ("\n<RECALL = %.2f>" % (sum(recalls_TP) / len(recalls_TP)))
    print ("\n<FALSE POSITIVE RATE = %.2f>" % (sum(false_positive_rates) / len(false_positive_rates)))
    print ("\n<F-SCORE = %.2f>" % ((2 * (sum(precisions_TP) / len(precisions_TP)) * (sum(recalls_TP) / len(recalls_TP))) / ((sum(precisions_TP) / len(precisions_TP)) + (sum(recalls_TP) / len(recalls_TP)))))


    #Searching for optimal number of iterations   
    training_errors_array = np.asarray(training_errors)
    min_error_index = np.argmin(training_errors_array)
    
    for error in training_errors_array:
        if error == 0:
            print ("\nThe 'ideal' number of iterations to terminate the training is:",min_error_index+1)
            break
    else:
        print ("\nThe 'ideal' number of iterations resulting in 0% training error has not been reached,")
        print ("increase the number of iterations to fine the 'ideal' training termination point.")
    
    print ("##############        END         ##############")
    
    eval_graphs(accuracies, precisions_TP, recalls_TP, false_positive_rates)
    x.error_graph()
    p.show()

    
    
    
