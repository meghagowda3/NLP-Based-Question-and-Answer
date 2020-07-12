import pandas as pd
import numpy as np
import os
import re
import operator
import string
from collections import Counter
import csv

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#nltk.download('words')
#nltk.download('punkt')
#nltk.download('stopwords')
words = set(nltk.corpus.words.words())

from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.similarities import Similarity, MatrixSimilarity

#============================================
if os.path.exists("Relevant_search.csv"):
  os.remove("Relevant_search.csv")
# checking if any prior files is present in working directory, if available delete it.
#Search1='ram of lenovo'
#Most_Related_N=5

def Key_Search(Keyword,Most_Related_N):
    
    Keyword=Keyword
    Search1 = re.sub(r"http\S+", "", Keyword) 
    Search1 = Search1.replace('.',' ') 
    Search1=re.sub("<!--?.*?-->","",Search1)
    Search1=re.sub("(\\d|\\W)+"," ",Search1)
    Search1 = re.sub(r'\s+',' ',re.sub(r'[^\w \s]','',Search1) ).lower()
    Search1 = word_tokenize(Search1) # tokenize for generating bag of words
    Search1= [w for w in Search1 if w not in set(stopwords.words('english'))]
    
    # After cleaning the given question keyword using text nlp, if keyword returns empty then given keywords will be assigned/used directly for analysis 
    if len(Search1)==0:
        Search1=word_tokenize(Keyword)
    
    for i in range(4):        
        if i == 0:
            df = pd.read_csv('data1.csv')
        elif i == 1:
            df = pd.read_csv('data2.csv')
        elif i == 2:
            df = pd.read_csv('data3.csv')            
        else:
            df = pd.read_csv('data4.csv')
           
        processed_docs=[]
        
        df.reset_index
        df.set_index('Question')
        df.isnull().sum()
        # Processing Keywords
        docs = df['Keys'].tolist()
        processed_docs = [word_tokenize(doc.lower()) for doc in docs] #tokenize and lowercase to generate bag of words
        # creating model for the data file        
        dictionary = Dictionary(processed_docs) # create a dictionary of words from our keywords
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs] #create corpus where the corpus is a bag of words for each document
        tfidf = TfidfModel(corpus) #create tfidf model of the corpus        
        sims = MatrixSimilarity(tfidf[corpus], num_features=len(dictionary)) # Create the similarity data structure. This is the most important part where we get the similarities between the Question -and Answers.
        # creating model for the keyword 
        query_doc_bow = dictionary.doc2bow(Search1) # get a bag of words from the query_doc
        query_doc_tfidf = tfidf[query_doc_bow] #convert the regular bag of words model to a tf-idf model of the keywords and it's tf-idf value for the question and answer
              
        similarity_array = sims[query_doc_tfidf] # get the array of similarity values between our new keywords and every other keywords. 
                    
        similarity_series = pd.Series(similarity_array.tolist(), index=df.Question.values) #Convert to a Series
        Most_Related = similarity_series.sort_values(ascending=False)[:Most_Related_N] #get the top matching results, 
        Most_Related.to_csv('Relevant_search.csv', mode='a',  header=False)
        del dictionary,corpus,tfidf, sims, query_doc_bow, query_doc_tfidf, similarity_array, similarity_series, Most_Related, processed_docs, docs
    
    col_Names=["Question", "Score"]
    df1 = pd.read_csv("Relevant_search.csv",names=col_Names)
    df1=df1.sort_values(by=['Score'], ascending=False)
    print('+'*25)  
    print('keyword: ', Keyword )
    print('--------&-------------')  
    print('Most Related Top ', Most_Related_N , ' Questions and Answers are as:')
    print('+'*25)  
    for j in range(Most_Related_N):
        Accuracy_Score=df1['Score'].loc[j]
        #print('Accuracy is ', Accuracy_Score)
        split_lines=''
        split_lines=df1['Question'].loc[j] 
        print(split_lines.split('\n', 1)[0])
        print(split_lines.split('\n', 1)[1])
        #Accuracy_Score=df1['Score'].loc[j]
        print('Accuracy is ', round(Accuracy_Score*100,2),'%')
        print('*'*25)    
    return

key_rec=Key_Search(input('Hello, Please enter the Keyword \n for an Answer you want to search for: '),5)



