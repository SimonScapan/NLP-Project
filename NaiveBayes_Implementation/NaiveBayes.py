import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import statistics 

def binary_class_infos(data,label_column_name,label_1,label_2):
    # group data by label
    grouped_data = data.groupby(label_column_name)

    # calculate lengths
    len_all = len(train_data)
    len_positive = len(grouped_data.get_group(label_1)) 
    len_negative = len(grouped_data.get_group(label_2))

    # calculate class probs
    prob_positive = len_positive/len_all
    prob_negative = len_negative/len_all
    
    return len_all, len_positive, len_negative, prob_positive, prob_negative

def calc_document_probabilities(document,positive_corpus,negative_corpus,len_positive,len_negative,threshold,smoothing,without):
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
    for word in document:
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
    # count the wordfrequency for both corpus
    for index, row in data.iterrows():
        words = row["News"]
        # check for class
        if row["Label"] == 1:
            # iterrate over document
            for word in words:
                # check if word is already in corpus, if so add 1 if not add the word with value 1
                if word in positive_corpus:
                    positive_corpus[word]+=1
                else:
                    positive_corpus[word]=1
        else:
            # iterrate over document
            for word in words:
                # check if word is already in corpus, if so add 1 if not add the word with value 1
                if word in negative_corpus:
                    negative_corpus[word]+=1
                else:
                    negative_corpus[word]=1

    return positive_corpus, negative_corpus

def calculate_accuracy(predicted_labels, labels):
    fp=0
    tp=0
    fn=0
    tn=0

    # Calculated tp, tn, fp and fn based on predicted_labels and correct labels
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
    # Calculated accuracy based on tp, tn, fp and fn 
    return (tp+tn)/(fp+tp+fn+tn), tp, tn, fp, fn

def eval_naive_bayes(data,positive_corpus,negative_corpus, smoothing, without):
    # get data informations
    len_all, len_positive, len_negative, prob_positive, prob_negative = binary_class_infos(train_data,"Label",1,-1)
    
    threshold_values=[]
    accuracy_values=[]
    prediction_results=[]
    # use different thresholds
    for threshold in np.arange(0.0001,0.1,0.0001):
        # if we use words that are not in the train corpus, we need to set the threshold to 0
        if not without:
            threshold = 0
        labels=[]
        predicted_labels=[]
        # save used threshold
        threshold_values.append(threshold)
        # iterrate over test_data
        for index, row in test_data.iterrows():
            words = row["News"]
            label = row["Label"]
            # Calculate probabilities for all words and both classes
            positive_document_prob,negative_document_prob = calc_document_probabilities(words,positive_corpus,negative_corpus,len_positive,len_negative,threshold,smoothing,without)
            # Sumup the results
            results={1:positive_document_prob*prob_positive,-1:negative_document_prob*prob_negative}
            # Get predicted label by highest probability and add actual label for comparison
            predicted_labels.append(max(results, key=results.get))
            labels.append(label)
        # calculate accuracy and save results

        accuracy, tp, tn, fp, fn = calculate_accuracy(predicted_labels,labels)
        accuracy_values.append(accuracy)
        prediction_results.append({"tp":tp,"tn":tn,"fn":fn,"fp":fp})
        
        # if we don't use the threshold, we dont need to itterate over different thresholds.
        if not without:
            break
        
    return threshold_values, accuracy_values, prediction_results

def generate_summary(thresholds, accuracy, results, test_data_len, add_to_filename=""):
    # get index of best accuracy
    best_accuracy = max(accuracy)
    best_index = accuracy.index(best_accuracy)
    # print best threshold
    print(thresholds[best_index])
    # generate output document
    generate_result_document(test_data_len,best_accuracy,results[best_index]["tp"],results[best_index]["tn"],results[best_index]["fp"],results[best_index]["fn"],add_to_filename)
    
def generate_result_document(test_data_len,accuracy,tp,tn,fp,fn,add_to_filename=""):
    # create outputcontent
    output = f"""
    |--------------------------------------------------------------
    | the overall amount of data ist: {test_data_len}
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
    # save to file
    filename =  f"./output/output_{add_to_filename}_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    outputfile = open(filename,"w")
    outputfile.write(output) 
    outputfile.close()

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
print("Ohne Smoothing und mit berücksichtigung nicht vorhandener Wörter im Trainingskorpus")
generate_summary(thresholds1, accuracy1, results1,test_data_len,"WithoutSmoothAndDropping")
print("Mit Smoothing und mit berücksichtigung nicht vorhandener Wörter im Trainingskorpus")
generate_summary(thresholds2, accuracy2, results2,test_data_len,"WithSmoothAndWithoutDropping")
print("Ohne Smoothing und ohne nicht vorhandene Wörter im Trainingskorpus")
generate_summary(thresholds3, accuracy3, results3,test_data_len,"WithoutSmoothAndWithDropping")
print("Mit Smoothing und ohne nicht vorhandene Wörter im Trainingskorpus")
generate_summary(thresholds4, accuracy4, results4,test_data_len,"WithSmoothAndDropping")