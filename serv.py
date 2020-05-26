from flask import Flask, request
import requests
from twilio.twiml.messaging_response import MessagingResponse
import pandas as pd
import numpy as np
import pickle
import operator
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.metrics.pairwise import cosine_similarity
import random
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import warnings
warnings.filterwarnings("ignore")
import sys
from multiprocessing import Process
import time



stemmer = LancasterStemmer()
def cleanup(sentence):
    word_tok = nltk.word_tokenize(sentence)
    stemmed_words = [stemmer.stem(w) for w in word_tok]

    return ' '.join(stemmed_words)


le = LE()

tfv = TfidfVectorizer(min_df=1, stop_words='english')

data = pd.read_csv('add.csv')
questions = data['Question'].values

X = []
for question in questions:
    X.append(cleanup(question))

tfv.fit(X)
le.fit(data['Class'])

X = tfv.transform(X)
y = le.transform(data['Class'])


trainx, testx, trainy, testy = tts(X, y, test_size=.25, random_state=42)

model = SVC(kernel='linear')
model.fit(trainx, trainy)
print("SVC:", model.score(testx, testy))


def get_max5(arr):
    ixarr = []
    for ix, el in enumerate(arr):
        ixarr.append((el, ix))
    ixarr.sort()

    ixs = []
    for i in ixarr[-1:]:
        ixs.append(i[1])

    return ixs[::-1]


app = Flask(__name__)


@app.route('/bot', methods=['Get','POST'])
def bot():
    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()
    responded = False
    cnt = 0
    DEBUG = False
    TOP5 = False
    usr = incoming_msg
    if usr == 'hi':
       TOP5 = True
       #print("Will display most relevent result now")
    t_usr = tfv.transform([cleanup(usr.strip().lower())])
    class_ = le.inverse_transform(model.predict(t_usr)[0])
    questionset = data[data['Class']==class_]
    cos_sims = []
    for question in questionset['Question']:
       sims = cosine_similarity(tfv.transform([question]), t_usr)
       cos_sims.append(sims)
            
    ind = cos_sims.index(max(cos_sims))
    if not TOP5:
       str1 = "Bot: " + data['Answer'][questionset.index[ind]]
       msg.body(str1)
       responded =True
    return str(resp)   
    
    #print("\n"*2)
    '''if 'quote' in incoming_msg:
        # return a quote
        r = requests.get('https://api.quotable.io/random')
        if r.status_code == 200:
            data = r.json()
            quote = f'{data["content"]} ({data["author"]})'
        else:
            quote = 'I could not retrieve a quote at this time, sorry.'
        msg.body(quote)
        responded = True
    if 'cat' in incoming_msg:
        # return a cat pic
        msg.media('https://cataas.com/cat')
        responded = True
    if not responded:
        msg.body('I only know about famous quotes and cats, sorry!')
    return str(resp)'''


if __name__ == '__main__':
    #app.run()
    action_process = Process(target=app.run)
 
    # We start the process and we block for 5 seconds.
    action_process.start()
    action_process.join(timeout=150)
 
    # We terminate the process.
    action_process.terminate()
    print("Hey there! I timed out! You can do things after me!")
