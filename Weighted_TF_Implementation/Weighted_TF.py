import pandas as pd
import math
import pickle
import time


def get_tf(message, label):
      
    # empty Term Frequency Dictionary
    tf_dict = {}
    
    for word in message:
        # check if word is NOT in dict, than add
        if word not in tf_dict:
            # with Label the  TF get's his special weight
            tf_dict[word] = (message.count(word)/len(message)) * int(label)
        
    return tf_dict

def get_idf(data):

    # empty IDF Dictionary
    idf_dict = {}

    # initialize dictionary with all unique words from all entries
    for index, row in data.iterrows():
        for word in row[3]:
            if word not in idf_dict:
                idf_dict[word] = 0
    
    # calculate the Inverse Document Frequency with each word 
    # get amount of rows in DF
    doc_count = data.shape[0]

    for word in idf_dict:
        count = 0
        for index, row in data.iterrows():
            if word in row[3]:
                count += 1
        idf_dict[word] = math.log(doc_count / count)
    
    return idf_dict

def calc_wdf(data):
    # produce IDF with given data
    idf_dict = get_idf(data)
    tf_comb = {}

    # iter over rows in data
    for index, row in data.iterrows():
        # build TF for each row
        tf_dict = get_tf(row[3],row[2])
        for word in tf_dict:
            if word not in tf_comb:
                tf_comb[word] = [ tf_dict.get(word), 1 ]
            else:
                tf_comb[word] = [ tf_comb.get(word)[0] + tf_dict.get(word), tf_comb.get(word)[1] + 1 ]

    for word in idf_dict:
        # calculate Weighted Document Frequency
        idf_dict[word] = idf_dict.get(word) * ( tf_comb.get(word)[0] / tf_comb.get(word)[1] )
                
    return idf_dict

def get_prediction(wdf, news):
    # calculate the Term Frequency of news
    news_tf = get_tf(news, 1)
    prediction = 0

    # iter each word in list news
    for word in news:
        # check wether the word is in the Weighted Document Frequency
        if word in wdf:
            # calculate predicion
            prediction += wdf[word] * news_tf[word]

    # check if prediction is positive or negative and then assign to result
    if prediction >= 0:
        return 1
    elif prediction < 0:
        return -1

def create_save_wdf(data, length=None):

    if length != None:
        data = data[:length]
    
    start_time = time.time()
    wdf = calc_wdf(data)
    required_time = (time.time() - start_time)/60

    output = f"""
    |--------------------------------------------------------------
    | the required time for the learning is: {required_time} minutes
    |--------------------------------------------------------------
    """

    print(output)
    
    filename =  "./output/output_{}.txt".format(time.strftime('%Y%m%d-%H%M%S'))
    outputfile = open(filename,"w")
    outputfile.write(output) 
    outputfile.close()

    # safe wdf to a pickle file
    pickle.dump(wdf, open('wdf.pkl', 'wb'))

def load_wdf():

    return pickle.load( open('wdf.pkl', 'rb') )

def test_model(wdf, testdata):
    # initialize accuracy terms
    fp = 0
    tp = 0
    fn = 0
    tn = 0

    for index, row in testdata.iterrows():
        news = row[3]
        original_label = row[2]
        prediction_label = get_prediction(wdf, news)

        if original_label == -1 and prediction_label == -1:
            tn += 1
        elif original_label == 1 and prediction_label == 1:
            tp += 1
        elif original_label == -1 and prediction_label == 1:
            fp += 1
        elif original_label == 1 and prediction_label == -1:
            fn += 1


    data_amount = str(testdata.shape[0])
    accuracy = str((tp+tn)/(tp+tn+fp+fn))

    output = f"""
    |--------------------------------------------------------------
    | the overall amount of data ist: {data_amount}
    |--------------------------------------------------------------
    | the true positive rate is:      {tp}
    | the true negative rate is:      {tn}
    | the false positive rate is:     {fp}
    | the false negative rate is:     {fn}
    |--------------------------------------------------------------
    | the accuracy is:                {accuracy}
    |--------------------------------------------------------------
    """

    print(output)

    filename =  "./output/output_{}.txt".format(time.strftime('%Y%m%d-%H%M%S'))
    outputfile = open(filename,"w")
    outputfile.write(output) 
    outputfile.close()


#####################################
# train Weighted Document Frequency #
#####################################

# # Get Data from Preprocessing
# train_data = pd.read_pickle('./../Data_Preprocessing/train.pkl')

# # create wdf
# create_save_wdf(train_data)

####################################
# test Weighted Document Frequency #
####################################

# Get Data from Preprocessing
test_data = pd.read_pickle('./../Data_Preprocessing/test.pkl')

# load wdf
wdf = load_wdf()

test_model(wdf, test_data)
