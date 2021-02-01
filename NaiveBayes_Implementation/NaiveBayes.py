import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import statistics 

def dicts_without_duplication(dict1,dict2):
    for word in list(dict1.keys()):
        if word in dict2:
            dict2.pop(word)
            dict1.pop(word)

    for word in list(dict2.keys()):
        if word in dict1:
            dict2.pop(word)
            dict1.pop(word)
    return dict1, dict2

def binary_class_infos(data,label_column_name,label_1,label_2):
    grouped_data= data.groupby(label_column_name)

    len_all = len(train_data)
    len_positive = len(grouped_data.get_group(label_1)) 
    len_negative = len(grouped_data.get_group(label_2))

    prob_positive = len_positive/len_all
    prob_negative = len_negative/len_all
    
    return len_all, len_positive, len_negative, prob_positive, prob_negative

def calc_document_probabilities(words,positive_corpus,negative_corpus,len_positive,len_negative,threshold,smoothing,without):
    positive_weights=[]
    negative_weights=[]
    
    #Laplace smoothing
    if smoothing:
        alpha=1
        beta=1
    else:
        alpha=0
        beta=0

    # Go trough all words of the given document
    for word in words:
        # reset prob
        prob=0
        if word in positive_corpus:
            # calculate the probability for word
            prob = (positive_corpus[word]+alpha)/(len_positive+beta)
        elif not without:
            # calculate the probability if word not in trainingscorpus
            prob = alpha/(len_positive+beta)
        # check if the probability is higher than the threshold and append to the weight list
        if prob > threshold: 
            positive_weights.append(prob)
        # reset prob
        prob=0
        if word in negative_corpus:
            # calculate the probability for word
            prob = (negative_corpus[word]+alpha)/(len_negative+beta)
        elif not without:
            # calculate the probability if word not in trainingscorpus
            prob = alpha/(len_negative+beta)
        # check if the probability is higher than the threshold and append to the weight list
        if prob > threshold:
            negative_weights.append(prob)
    # calculate the product of all weights in both weight lists
    return np.prod(positive_weights),np.prod(negative_weights)

def train_naive_bayes(data):
    positive_corpus={}
    negative_corpus={}
    for index, row in data.iterrows():
        words = row["News"]
        if row["Label"] == 1:
            for word in words:
                if word in positive_corpus:
                    positive_corpus[word]+=1
                else:
                    positive_corpus[word]=1
        else:
            for word in words:
                if word in negative_corpus:
                    negative_corpus[word]+=1
                else:
                    negative_corpus[word]=1

    # print(f"Mean count of positive_corpus words in training {statistics.mean(positive_corpus.values())}")
    # print(f"Mean count of negative_corpus words in training {statistics.mean(negative_corpus.values())}")

    return positive_corpus, negative_corpus

def calculate_accuracy(predicted_labels, labels):
    fp=0
    tp=0
    fn=0
    tn=0

    # Calculated accuracy based on predicted_labels
    for real, pred in zip(labels, predicted_labels):
        if real == 1:
            if pred == 1:
                tp +=1
            elif pred == -1:
                fn +=1
        elif real == -1:
            if pred == 1:
                fp +=1
            elif pred == -1:
                tn +=1
    return (tp+tn)/(fp+tp+fn+tn), tp, tn, fp, fn

def eval_naive_bayes(data,positive_corpus,negative_corpus, smoothing, without):
    counter =0
    len_all, len_positive, len_negative, prob_positive, prob_negative = binary_class_infos(train_data,"Label",1,-1)
    
    threshold_values=[]
    accuracy_values=[]
    prediction_results=[]
    for threshold in np.arange(0.0001,0.1,0.0001):
        if not without:
            threshold = 0
        labels=[]
        predicted_labels=[]
        threshold_values.append(threshold)
        for index, row in test_data.iterrows():
            words = row["News"]
            label = row["Label"]
            # Calculate probabilities for all words and both classes
            positive_result,negative_result = calc_document_probabilities(words,positive_corpus,negative_corpus,len_positive,len_negative,threshold,smoothing,without)
            # Sumup the results
            results={1:positive_result*prob_positive,-1:negative_result*prob_negative}
            # Get predicted label by highest probability and add actual label for comparison
            predicted_labels.append(max(results, key=results.get))
            labels.append(label)
        accuracy, tp, tn, fp, fn = calculate_accuracy(predicted_labels,labels)
        accuracy_values.append(accuracy)
        prediction_results.append({"tp":tp,"tn":tn,"fn":fn,"fp":fp})
        if not without:
            break
        # elif counter == 2:
        #     break
        # counter+=1
        
        
    return threshold_values, accuracy_values, prediction_results

def generate_summary(thresholds, accuracy, results, test_data_len, add_to_filename=""):
    best_index = accuracy.index(max(accuracy))
    print(thresholds[best_index])
    generate_result_document(test_data_len,results[best_index]["tp"],results[best_index]["tn"],results[best_index]["fp"],results[best_index]["fn"],add_to_filename)
    
def generate_result_document(test_data_len,tp,tn,fp,fn,add_to_filename=""):
    data_amount = str(test_data_len)
    accuracy = str((tp+tn)/(tp+tn+fp+fn))

    output = f"""
    |--------------------------------------------------------------
    | the overall amount of data ist: {data_amount}
    |--------------------------------------------------------------
    | the true positive_corpus rate is:      {tp}
    | the true negative_corpus rate is:      {tn}
    | the false positive_corpus rate is:     {fp}
    | the false negative_corpus rate is:     {fn}
    |--------------------------------------------------------------
    | the accuracy is:                {accuracy}
    |--------------------------------------------------------------
    """

    print(output)
    filename =  f"./output/output_{add_to_filename}_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    outputfile = open(filename,"w")
    outputfile.write(output) 
    outputfile.close()

def plot_accuracy(x,y):
    plt.plot(x,y)
    plt.ylabel('accuracy.')
    plt.show()

################################################################################################

# Get Data from preprocessing
train_data = pd.read_pickle('../Data_Preprocessing/train.pkl')
test_data = pd.read_pickle('../Data_Preprocessing/test.pkl')

# train naive_bayes
positive_corpus, negative_corpus = train_naive_bayes(train_data)
test_data_len = len(test_data)

# evaluate naive bayes with different settings
thresholds1, accuracy1, results1 = eval_naive_bayes(test_data,positive_corpus, negative_corpus, smoothing=False, without=False)
thresholds2, accuracy2, results2 = eval_naive_bayes(test_data,positive_corpus, negative_corpus, smoothing=True, without=False)
thresholds3, accuracy3, results3 = eval_naive_bayes(test_data,positive_corpus, negative_corpus, smoothing=False, without=True)
thresholds4, accuracy4, results4 = eval_naive_bayes(test_data,positive_corpus, negative_corpus, smoothing=True, without=True)

# show evaluationresults and generate outputfiles
print("Ohne Smoothing und berücksichtigung nicht vorhandener Wörter im Trainingskorpus")
generate_summary(thresholds1, accuracy1, results1,test_data_len,"WithoutSmoothAndDropping")
print("Mit Smoothing und berücksichtigung nicht vorhandener Wörter im Trainingskorpus")
generate_summary(thresholds2, accuracy2, results2,test_data_len,"WithSmoothAndWithoutDropping")
print("Ohne Smoothing und nicht vorhandene Wörter im Trainingskorpus")
generate_summary(thresholds3, accuracy3, results3,test_data_len,"WithoutSmoothAndWithDropping")
print("Mit Smoothing und ohne nicht vorhandene Wörter im Trainingskorpus")
generate_summary(thresholds4, accuracy4, results4,test_data_len,"WithSmoothAndDropping")