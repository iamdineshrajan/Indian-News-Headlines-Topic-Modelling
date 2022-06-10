# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:47:18 2022

@author: DELL
"""

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import regex as re
import string

def TextPreprocessing(text):
    text = re.sub(r'[^a-z\s]', '', str(text).lower().strip())
    text = word_tokenize(text)
    stop_words = stopwords.words('english')
    text = [word for word in text if word not in stop_words]
    word_lem = WordNetLemmatizer()
    news_text = [word_lem.lemmatize(word,pos='v') for word in text]
    return news_text

