import pandas as pd
from django.shortcuts import render
import os
import pickle
from math import log10, sqrt
from transformers import pipeline, RobertaTokenizerFast, RobertaForSequenceClassification
import tensorflow.compat.v1 as tf
import warnings
import ast


warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated.*")

# Suppress deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Define stoplist globally
current_dir = os.path.dirname(__file__)
stopwords_file = os.path.join(current_dir, 'collection', 'stopwords.txt')
with open(stopwords_file, 'r') as x:
    stoplist = set(x.read().lower().split())

def preprocess_text(text):
    return ' '.join([word.lower() for word in text.split() if len(word) > 1 and word.lower() not in stoplist])

def preprocess_text2(text):
    return ' '.join([word.lower() for word in text.split() if len(word) > 1 and word.lower()])

tokenaser = RobertaTokenizerFast.from_pretrained('arpanghoshal/EmoRoBERTa')
model = RobertaForSequenceClassification.from_pretrained('arpanghoshal/EmoRoBERTa', from_tf=True)


def get_word_emotions(words):
    emotions = []
    for word in words:
        emotion = pipeline('sentiment-analysis', model=model, tokenizer=tokenaser)
        emotions.append(emotion(word))
    return emotions
    
def get_query_emotion(search_query):
    words = preprocess_text(search_query).split()
    word_emotions = get_word_emotions(words)  
    emotion_counts = {} 
    for emotion in word_emotions:    # Count the occurrences of each emotion
        emotion_label = emotion[0]['label']
        emotion_counts[emotion_label] = emotion_counts.get(emotion_label, 0) + 1
    max_count = 0
    most_common_emotion = None
    most_common_score = None
    for emotion, count in emotion_counts.items():
        if count > max_count:
            max_count = count
            most_common_emotion = emotion
            most_common_score = None
            for em in word_emotions:    # Iterate through emotions again to find the score of the most common emotion
                if em[0]['label'] == emotion:
                    most_common_score = em[0]['score']
                    break
    return most_common_emotion, most_common_score


def index(request):
    N = 5332
    freq = {}
    data = pd.read_csv(os.path.join(current_dir, 'collection', 'netflix_titles_processed.csv'))
    # Calculate word frequency and max frequency per document
    for k in range(1, N + 1):
        description = preprocess_text(data.loc[k, 'description'])
        title = preprocess_text(data.loc[k, 'title'])

        words = description.split() + title.split()
        # Calculate the frequency of each word in the document
        for w in set(words):
            freq[(w, k)] = words.count(w)

    # Calculate ni (number of documents containing each word)
    ni = {}
    for (w, d) in freq:
        ni[w] = ni.get(w, 0) + 1

    # Calculate max frequency of any word for each document
    max_freq = {}
    for (w, d) in freq:
        max_freq[d] = max(max_freq.get(d, 0), freq[(w, d)])

    # Calculate TF-IDF weights and store them in the index along with document emotions
    poids = {}
    for (w, d), f in freq.items():
        tfidf_weight = (f / max_freq[d]) * log10((N / ni[w]) + 1)
        # Store TF-IDF weight
        poids[(w, d)] = tfidf_weight


    # Save TF-IDF weights to a pickle file
    with open(os.path.join(current_dir, 'collection', 'MonIndex.pkl'), "wb") as tf:
        pickle.dump(poids, tf)

    return render(request, 'index.html')


def search(request):

    with open(os.path.join(current_dir, 'collection', 'MonIndex.pkl'), "rb") as tf_file:
        poids = pickle.load(tf_file)

    N = 5332  
    ni = {}
    for (w, d) in poids:
        ni[w] = ni.get(w, 0) + 1

    def produit_Interne(poids, q):
        rsv = {}
        q = preprocess_text(q)
        qT = q.split()

        for i in range(N):
            rsv[i+1] = 0

        for (t, d) in poids:
            if t in qT and t not in stoplist:
                rsv[d] = rsv[d] + poids[t, d]

        return rsv

    def cosinus(poids, q):
        rsv = {}
        q = preprocess_text(q)
        qT = q.split()
        som = {}
        SQT = sum(1 for t in qT if t in ni)

        if SQT == 0:
            for i in range(N):
                rsv[i+1] = 0
        else:
            rsv = produit_Interne(poids, q)
            for (t, d) in poids:
                som[d] = som.get(d, 0) + (poids[t, d] * poids[t, d])

            for d in rsv:
                rsv[d] = rsv[d] / sqrt(som[d] * SQT)

        return rsv

    if request.method == 'GET' and 'search_query' in request.GET:
        search_query = request.GET.get('search_query', '')
        
        query_emotion,query_emotion_score = get_query_emotion(search_query)
        rsv = cosinus(poids, search_query)
        search_results = []
        data = pd.read_csv(os.path.join(current_dir, 'collection', 'netflix_titles_processed.csv'))
            
        for doc, score in rsv.items():
            if preprocess_text2(search_query) == preprocess_text2(data.loc[doc,"title"]):
                rsv[doc]=1

            # Get the emotion string from the DataFrame
            doc_emotion_str = data.loc[doc, 'emotion']
            
            # Safely evaluate the string to convert it into a list of dictionaries
            doc_emotion_list = ast.literal_eval(doc_emotion_str)
            
            # Initialize doc_emotion_score
            doc_emotion_score = None
            
            # Iterate through the list of dictionaries to find the 'score' value
            for emotion_dict in doc_emotion_list:
                
                if 'label' in emotion_dict:
                    if emotion_dict['label']==query_emotion and emotion_dict['label']!="neutral":
                        doc_emotion_score = emotion_dict['score']
                    break
            
            # Update the relevance score by query_emotion_score
            if doc_emotion_score is not None:
                rsv[doc] = (score * 0.6) + (doc_emotion_score * 0.4)
        sorted_results = sorted(rsv.items(), key=lambda item: item[1], reverse=True)     

        for doc, score in sorted_results:
            doc_name = data.loc[doc, 'title']
            doc_type=data.loc[doc,'type']
            doc_date_added=data.loc[doc,'date_added']
            doc_release_year=data.loc[doc,'release_year']
            doc_rating=data.loc[doc,'rating']
            doc_duration=data.loc[doc,'duration']
            doc_country=data.loc[doc,'country']
            doc_director=data.loc[doc,'director']
            doc_cast=data.loc[doc,'cast']
            doc_listed_in=data.loc[doc,'listed_in']
            document_content = data.loc[doc, 'description']
            if score != 0:
                search_results.append({'nom_doc': doc_name, 'document_content': document_content, 'score': score,
                                       'document_type':doc_type,'document_date_added':doc_date_added,
                                       'document_release_year':doc_release_year,'document_rating':doc_rating,
                                       'document_duration':doc_duration,'document_country':doc_country,
                                       'document_director':doc_director,
                                       'document_cast':doc_cast,'document_listed_in':doc_listed_in})
    else:
        msg = "You need to type a query first!"
        return render(request, 'index.html', {'message': msg})
    return render(request, 'search.html', {'search_query': search_query, 'search_results': search_results, 'query_emotion': query_emotion,
                                           'query_emotion_score':query_emotion_score})
