import os
import operator
import glob
from nltk.tokenize import word_tokenize
import numpy as np
from stop_words import get_stop_words
import string
from pymorphy2 import MorphAnalyzer
morph = MorphAnalyzer()
from math import log
from collections import Counter


def open_():
    path = 'D://avito/*.txt'  # note C:
    files = glob.glob(path)
    N = len(files)
    texts = []
    for name in files:
        with open(name, encoding='utf-8') as file:
            f = file.read()
            f = f.replace('\xa0', ' ')
            texts.append(f)
    sum_len = 0
    for text in texts:
        sum_len += len(text)
    av_len = float(len(texts) / sum_len)
    return texts, N, av_len


def preprocessing(input_text, del_stopwords=True, del_digit=True):
    russian_stopwords = set(get_stop_words('russian'))
    words = [x.lower().strip(string.punctuation + '»«–…') for x in word_tokenize(input_text)]
    lemmas = [morph.parse(x)[0].normal_form for x in words if x]
    lemmas_arr = []
    for lemma in lemmas:
        if del_stopwords:
            if lemma in russian_stopwords:
                continue
        if del_digit:
            if lemma.isdigit():
                continue
        # pos = morph.parse(lemma)[0].tag.POS
        # lemma = lemma + '_' + str(pos)
        lemmas_arr.append(lemma)
    #     lemmas_ = ' '.join(lemmas_arr)
    return lemmas_arr


texts, N, av_len = open_()
texts_ = []
for text in texts:
    prepr = preprocessing(text, del_stopwords=True, del_digit=True)
    texts_.append(prepr)
# print(texts_)


def freq(query, texts_, texts):
    query = query.split()
    counts = {}
    for text_, text in zip(texts_, texts):
        # text_ = preprocessing(text, del_stopwords=True, del_digit=True)
        text_len = len(text_)
        for quer in query:
            count = Counter(text_)[quer]
            counts[text] = [count, text_len]
    # n = 0
    # for key, value in counts.items():
    #     dl = value[1]
    #     qf = value[0]
    # print(counts.items())
    return counts



k1 = 2.0
b = 0.75


def score_BM25(qf, dl, avgdl, k1, b, N, n, counts) -> float:
    score = log((N - n + 0.5) / (n + 0.5)) * (k1 + 1) * qf / (qf + k1 * (1 - b + b * dl / avgdl))
    return score


def fin_ok(query, texts, N, av_len):
    q = freq(query, texts_, texts)
    n = 0
    for key, value in q.items():
        # print(key, value)
        if value[0] != 0:
            n += 1
        dl = value[1]
        qf = value[0]
        rev_score = score_BM25(qf, dl, av_len, k1, b, N, n, q)
        # print(rev_score)
        value.append(rev_score)
        value.remove(value[0])
        value.remove(value[0])
    scores = sorted(q, key=q.get, reverse=True)
    return scores[:5]
