import sys
from random import shuffle
import numpy
import BagOfWords_model
import Bernoulli_model
from sklearn import metrics
import matplotlib.pyplot as plt

'''
This function is used to calculate the posteriors using given weights\
    and training data instances
'''
def posteriorCalc(weights, train_data_instance):
    value = weights['zero_weight'] * 1
    for i in train_data_instance:
        if i == 'unique_category' or i == 'zero_weight':
            continue
        else:
            if i in weights:
                if i in train_data_instance:
                    value = value + (weights[i] * train_data_instance[i])
    x=float(1 + numpy.exp(-value))
    posterior=1/x
    return posterior

'''
    This function is used to train the logistic regression using train data, \
        total list of words in train set and tunes the weights
'''
def train_MCAP_LR(train_data, Vocab, eta, lamda, iterations):
    weights = dict(Vocab)
    for i in weights:
        weights[i] = 0
    weights['zero_weight'] = 0
    # Updating the weights
    for i in range(iterations):
        for dictionary in train_data:
            posterior = posteriorCalc(weights, dictionary)
            sum_values= 0
            for weight in weights:
                # Here I checked if the weight is not equal to 0 or not
                if dictionary[weight] != 0:
                    # This is the case when w_o is used
                    if weight == "zero_weight":
                        sum_values = sum_values + eta * (dictionary["unique_category"] - posterior)
                    else:
                        # This is the case when other w's are used
                        sum_values = sum_values + eta * (dictionary[weight] * (dictionary["unique_category"] - posterior))
                    weights[weight] = weights[weight] + sum_values - eta * lamda * weights[weight]
    return weights


#To predict the output class for the given test instance and tuned weights
def test_MCAP_LR(test_example, weights):
    value = weights['zero_weight'] * 1
    for i in test_example:
        if i == 'unique_category' or i == 'zero_weight':
            continue
        else:
            if i in weights and i in test_example:
                value = value + (weights[i] * test_example[i])
    # spam=1, ham=0
    if value < 0:
        return 0
    else:
        return 1



dataset = str(input('Enter dataset name(enron1/enron4/hw1): '))
print("\nEnter representing model for dataset:\n1 for Bag of words \n2 for Bernoulli: ")
representing_model=int(input("Enter 1 or 2: "))

#load training data according to the model selected.
if (representing_model==1):
    spam_representation, ham_representation, Vocab_freq, spam_allwords_freq, ham_allwords_freq, no_of_totalfiles, no_of_spamfiles, no_of_hamfiles, Vocab = BagOfWords_model.BOW(dataset,'train')
elif(representing_model==2):
    spam_representation, ham_representation, spam_allwords_freq, ham_allwords_freq, no_of_totalfiles, no_of_spamfiles, no_of_hamfiles, Vocab = Bernoulli_model.bernoulli(dataset,'train')
else:
    print("Wrong input!!!")
    sys.exit()


# adding a key to the models to represent if it's ham or spam
for file in spam_representation:
    file["unique_category"] = 1
    file["zero_weight"] = 1
for file in ham_representation:
    file["unique_category"] = 0
    file["zero_weight"] = 1
complete_data = spam_representation + ham_representation

#shuffling and dividing data into 70-30 split (70-training, 30-validation)
shuffle(complete_data)
# 70 percent of the data is for traning and 30 percent of the data is for validation
train_data = complete_data[0: int(0.7*len(complete_data))]
validation_data = complete_data[int(0.7*len(complete_data)): -1]


accu_maximum = 0
best_lambda = 1
lis=[1000,100,10,1,0.1,0.01,0.001]
'''
Took the lambda values in the above list to tune it  
C=1/lambda
Generally if c is larger it better fits the data
'''
for lamda in lis:
    # We train the algo with the train data
    weights = train_MCAP_LR(train_data, Vocab, 0.01, lamda, 50)
    perfectly_classified = 0
    # We test on the validation data
    for doc in validation_data:
        output = test_MCAP_LR(doc, weights)
        if output == doc["unique_category"]:
            perfectly_classified = perfectly_classified + 1
    accuracy = perfectly_classified / float(len(validation_data))
    # Here we get the best lambda value
    if accu_maximum < accuracy:
        accu_maximum = accuracy
        best_lambda = lamda


lamda=best_lambda
train_data = train_data + validation_data
# to tune the weights
weights = train_MCAP_LR(train_data, Vocab, 0.01,lamda, 500)


if (representing_model==1):
    spam_representation_test, ham_representation_test, Vocab_freq, spam_allwords_freq, ham_allwords_freq, no_of_totalfiles, no_of_spamfiles, no_of_hamfiles, Vocab = BagOfWords_model.BOW(dataset,'test')
elif(representing_model==2):
    spam_representation_test, ham_representation_test, spam_allwords_freq, ham_allwords_freq, no_of_totalfiles, no_of_spamfiles, no_of_hamfiles, Vocab = Bernoulli_model.bernoulli(dataset,'test')


spam_predicted = []
spam_true=[]
# In this step the algorithm predicts the output for a given dataset
for each_document in spam_representation_test:
    spam_predicted.append(test_MCAP_LR(each_document, weights))
    spam_true.append(1)


ham_predicted = []
ham_true=[]
for each_document in ham_representation_test:
    ham_predicted.append(test_MCAP_LR(each_document, weights))
    ham_true.append(0)


y_true = spam_true + ham_true
y_predicted = spam_predicted + ham_predicted


confusion_matrix = metrics.confusion_matrix(y_true, y_predicted)

matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Ham', 'Spam'])

matrix_display.plot()
plt.show()

print ('\nAccuracy:', metrics.accuracy_score(y_true, y_predicted))
print ('Precision:', metrics.precision_score(y_true, y_predicted))
print ('Recall:', metrics.recall_score(y_true, y_predicted))
print ('f1 score:', metrics.f1_score(y_true, y_predicted))