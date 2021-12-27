import numpy as np
import pandas as pd
import os
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random
import opendatasets as od
from kaggle.models.dataset_upload_file import DatasetUploadFile  # noqa: F401,E501


"""
credit to Normalized Nerd ` https://www.youtube.com/watch?v=E4WcBWuQQws&list=PLM8wYQRetTxBkdvBtz-gw8b9lcVkdXQKV&index=4
"""

def read_all_stories(story_path):
    txt = []
    for _, _, files in os.walk(story_path):
        for file in files:
            with open(story_path+file) as f:
                for line in f:
                    line = line.strip()
                    if line=='----------': break
                    if line!='':txt.append(line)
    return txt
        


def clean_txt(txt):
    cleaned_txt = []
    for line in txt:
        line = line.lower()
        line = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-\\]", "", line)
        tokens = word_tokenize(line)
        words = [word for word in tokens if word.isalpha()]
        cleaned_txt+=words
    return cleaned_txt


def make_markov_model(cleaned_stories, n_gram=2):
    markov_model = {}
    for i in range(len(cleaned_stories)-n_gram-1):
        curr_state, next_state = "", ""
        for j in range(n_gram):
            curr_state += cleaned_stories[i+j] + " "
            next_state += cleaned_stories[i+j+n_gram] + " "
        curr_state = curr_state[:-1]
        next_state = next_state[:-1]
        if curr_state not in markov_model:
            markov_model[curr_state] = {next_state: 1}
        elif next_state in markov_model[curr_state]:
            markov_model[curr_state][next_state] += 1
        else:
            markov_model[curr_state][next_state] = 1

    # calculating transition probabilities
    for curr_state, transition in markov_model.items():
        total = sum(transition.values())
        for state, count in transition.items():
            markov_model[curr_state][state] = count/total

    return markov_model


def generate_story(markov_model, limit=100, start='my god'):
    n = 0
    curr_state = start
    next_state = None
    story = ""
    story+=curr_state+" "
    while n<limit:
        next_state = random.choices(list(markov_model[curr_state].keys()),
                                    list(markov_model[curr_state].values()))
        
        curr_state = next_state[0]
        story+=curr_state+" "
        n+=1
    return story


if __name__=='__main__':
    
    stories = read_all_stories(od.download("https://www.kaggle.com/<idangrady>/orion99/markov-chain-nlp/data"))
    print("number of lines = ", len(stories))

    print(stories[2:10])
    cleaned_stories = clean_txt(stories[2:10])
    print()
    print(cleaned_stories)
    print("number of words = ", len(cleaned_stories))
    
    markov_model = make_markov_model(cleaned_stories)
    print("number of states = ", len(markov_model.keys()))
    
    print(markov_model)