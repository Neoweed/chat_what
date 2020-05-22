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
from nltk.stem.lancaster import LancasterStemmer
import warnings
warnings.filterwarnings("ignore")
import sys

app = Flask(__name__)

stemmer = LancasterStemmer()
def cleanup(sentence):
    word_tok = nltk.word_tokenize(sentence)
    stemmed_words = [stemmer.stem(w) for w in word_tok]

    return ' '.join(stemmed_words)


le = LE()

tfv = TfidfVectorizer(min_df=1, stop_words='english')

data = pd.read_csv('FAQs.csv')
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


@app.route('/bot', methods=['POST'])
def bot():
    usr = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()
    responded = False
    cnt = 0

    #print("PRESS Q to QUIT")
    #print("TYPE \"DEBUG\" to Display Debugging statements.")
    #print("TYPE \"STOP\" to Stop Debugging statements.")
    #print("TYPE \"TOP\" to Display most relevent result")
    #print("TYPE \"CONF\" to Display the most confident result")
    #print()
    #print()
    DEBUG = False
    TOP5 = False

    msg.body("Bot: Hi, Welcome to our bank!")
    respond = True
    while True:
        usr = request.values.get('Body', '').lower()
        #usr = input("You: ")
        #usr='hi'
        if usr.lower() == 'yes':
            msg.body("Bot: Yes!")
            responded = True
            continue

        if usr.lower() == 'no':
            msg.body("Bot: No?")
            responded = True
            continue

        if usr == 'DEBUG':
            DEBUG = True
            msg.body("Debugging mode on")
            responded = True
            continue

        if usr == 'STOP':
            DEBUG = False
            msg.body("Debugging mode off")
            responded = True
            continue

        if usr == 'Q':
            msg.body("Bot: It was good to be of help.")
            responded = True
            break

        if usr == 'hi':
            TOP5 = True
            msg.body("Will display most relevent result now")
            responded = True
            continue

        if usr == 'CONF':
            TOP5 = False
            msg.body("Only the most relevent result will be displayed")
            responded = True
            continue

        t_usr = tfv.transform([cleanup(usr.strip().lower())])
        class_ = le.inverse_transform(model.predict(t_usr)[0])
        questionset = data[data['Class']==class_]

        if DEBUG:
            msg.body("Question classified under category:", class_)
            responded = True
            msg.body("{} Questions belong to this class".format(len(questionset)))
            responded = True

        cos_sims = []
        for question in questionset['Question']:
            sims = cosine_similarity(tfv.transform([question]), t_usr)
            cos_sims.append(sims)
            
        ind = cos_sims.index(max(cos_sims))

        if DEBUG:
            question = questionset["Question"][questionset.index[ind]]
            msg.body("Assuming you asked: {}".format(question))
            responded = True

        if not TOP5:
            msg.body("Bot:", data['Answer'][questionset.index[ind]])
            responded = True
        else:
            inds = get_max5(cos_sims)
            for ix in inds:
                msg.body("Question: "+data['Question'][questionset.index[ix]])
                msg.body("Answer: "+data['Answer'][questionset.index[ix]])
                msg.body('-'*50)

        #print("\n"*2)
        #usr = input("You:")
        """ outcome = input("Was this answer helpful? Yes/No: ").lower().strip()
        if outcome == 'yes':
            cnt = 0
        elif outcome == 'no':
            inds = get_max5(cos_sims)
            sugg_choice = input("Bot: Do you want me to suggest you questions ? Yes/No: ").lower()
            if sugg_choice == 'yes':
                q_cnt = 1
                for ix in inds:
                    print(q_cnt,"Question: "+data['Question'][questionset.index[ix]])
                    # print("Answer: "+data['Answer'][questionset.index[ix]])
                    print('-'*50)
                    q_cnt += 1
                num = int(input("Please enter the question number you find most relevant: "))
                print("Bot: ", data['Answer'][questionset.index[inds[num-1]]])
    if 'quote' in incoming_msg:
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
    return str(resp)
"""

if __name__ == '__main__':
    app.run()
