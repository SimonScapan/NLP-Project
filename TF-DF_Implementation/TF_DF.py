import pandas as pd
import math
import pickle
import time


def get_tf(Message, Label):
        
    # empty Term Frequency Dictionary
    TFdict = {}
    
    for word in Message:
        
        # caount amount of words
        wordcount = len(Message)

        # check if word is NOT in dict, than add
        if word not in TFdict:

            # count amount of same words
            amount = Message.count(word)
            TF = (amount/wordcount) * int(Label)
            TFdict.update({word:TF})

        else: 

            # if a word is allready in dict then pass
            pass
        
    # return the TF dictionary
    return TFdict


def get_idf(data):

    # empty IDF Dictionary
    IDFdict = {}

    # fill dictionary with all unique words from all entries
    for index, row in data.iterrows():
        for word in row[3]:

            if word not in IDFdict:

                IDFdict.update({word:0})
    
    # calculate the Inverse Document Frequency of each word
    doc_count = data.shape[0]

    for word in IDFdict:

        count = 0

        for index, row in data.iterrows():
            print('IDF row: ' + str(row[0]))
            if word in row:

                count += 1
            else:
                count = 1

        idf = math.log(doc_count / count)
        IDFdict[word] = idf
    
    # return the IDF Dictionary
    return IDFdict


def calc_wdf(data):

    # produce IDF with given data
    IDFdict = get_idf(data)

    # iter over rows in data
    for index, row in data.iterrows():
        print('new row, time in calc wdf: ' + str(time.time()))
        # build TF for each row
        TFdict = get_tf(row[3],row[2])

        for word in TFdict:
            
            if word in IDFdict:

                idf_val = IDFdict.get(word)
                tf_val = TFdict.get(word)
                wdf_val = idf_val * tf_val

                # calculate Weighted Document Frequency
                IDFdict[word] = wdf_val
    
    return IDFdict


def call(WDF, news):

    # split news into a list of words
    news = news.split()

    # calculate the Term Frequency of news
    news_tf = get_tf(news, 1)

    prediction = 0

    # iter each word in list news
    for word in news:

        # check wether the word is in the Weighted Document Frequency
        if word in WDF:
            
            # calculate predicion
            prediction += WDF[word] * news_tf[word]
        
        else:
            pass

    # check if prediction is positive or negative and then assign to result
    if prediction >= 0:
        return 'positive'

    elif prediction < 0:
        return 'negative'


def create_save_wdf(data, length=None):

    if length != None:
        data = data[:length]
    
    start_time = time.time()
    WDF = calc_wdf(data)
    required_time = (time.time() - start_time)/60
    print("--- %s minutes ---" %required_time )

    # safe WDF to a pickle file
    pickle.dump(WDF, open('./TF-DF_Implementation/WDF.pkl', 'wb'))


def load_wdf():

    return pickle.load( open('./TF-DF_Implementation/WDF.pkl', 'rb') )



####################################
# call Weighted Document Frequency #
####################################

# Get Data from Preprocessing
data = pd.read_pickle('./Data_Preprocessing/train.pkl')

# create wdf
create_save_wdf(data)

# # load wdf
# wdf = load_wdf()

# # create news text
# news = 'Today is a good day'

# # call prediction
# prediction = call(wdf, news)
# print(prediction)


