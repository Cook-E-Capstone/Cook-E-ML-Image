# Import Dependencies
import pandas as pd
import numpy as np
import os
import requests, re, string
import difflib
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory as SF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Read dataset to get the result
df_resep = pd.read_csv('app/data/resep_final.csv')

# Read dataset for processing
df_cleaned = pd.read_csv('app/data/df_cleaned.csv')
df_cleaned['tags'] = df_cleaned['menu'] + ' ' + df_cleaned['bahan'] + ' ' + df_cleaned['langkah']

# Make object Tfidf and Cosine Similarity for searching the desired resep
vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
features_vectors = vectorizer.fit_transform(df_cleaned['tags'])
similarity = cosine_similarity(features_vectors)

# Function to clean the user input
def case_folding(data):
    data = data.lower()
    data = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",data).split())
    data = re.sub(r"\d+", "", data)
    data = data.translate(str.maketrans("","",string.punctuation))
    data = re.sub(r"\n","",data)
    data = re.sub(r"\t","",data)
    return data

def stopword_cleaner(data):
    sw_indonesia = stopwords.words("indonesian")
    sw_indonesia.remove('tahu')
    data  = [word for word in data if word not in sw_indonesia]
    data = ' '.join(data)
    return data

# Function to process user input and get the result
def recommend(keywords, limit):
    # Search the input that quite similar to dataset 
    similarity_scores = [difflib.SequenceMatcher(None, keywords, tags).ratio() for tags in df_cleaned['bahan'].tolist()]
    closest_match_index = similarity_scores.index(max(similarity_scores))
    similarity_score = sorted(list(enumerate(similarity[closest_match_index])), key=lambda x:x[1], reverse=True)

    # Create an empty list to store the dictionaries
    result_list = []
    
    # Append each row of df_result as a dictionary to the list
    i = 1
    for item in similarity_score:
        index = item[0]
        val = item[1]
        if i <= limit:
            temp_data = (df_resep.loc[index].to_dict()).copy()
            temp_data["bahan"] = ast.literal_eval(temp_data["bahan"])
            temp_data["langkah"] = ast.literal_eval(temp_data["langkah"])
            result_list.append(temp_data)
            i += 1
    
    return result_list

# # Ini return nama-nama menunya doang (opsional/kalo butuh)
# def rekomendasi(keywords):
#     # Search the input that quite similar to dataset 
#     similarity_scores = [difflib.SequenceMatcher(None, keywords, tags).ratio() for tags in df_cleaned['bahan'].tolist()]
#     closest_match_index = similarity_scores.index(max(similarity_scores))
#     similarity_score = sorted(list(enumerate(similarity[closest_match_index])), key=lambda x:x[1], reverse=True)
    
#     # Return the data (ini format dataframe)
#     list_menu = []
#     i = 1
#     for item in similarity_score:
#         index = item[0]
#         menu_from_index = df_cleaned[df_cleaned.index==index]['menu'].values[0]
#         if(i<11):
#             list_menu.append(menu_from_index)
#             i+=1
#     return list_menu

# rekomendasi(kata_kunci)

