import gensim
from nltk.tokenize import word_tokenize
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from gensim import matutils
from stop_words import get_stop_words
import string
from pymorphy2 import MorphAnalyzer
morph = MorphAnalyzer()
import os
import operator
import glob
base = 'https://www.avito.ru/moskva/odezhda_obuv_aksessuary/'
# model_path = 'D:/tayga_1_2.vec'
# model = KeyedVectors.load_word2vec_format(model_path, binary=False)
model_path = 'D:/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
model = Word2Vec.load(model_path)


def similarity(v1, v2):
    v1_norm = matutils.unitvec(np.array(v1))
    v2_norm = matutils.unitvec(np.array(v2))
    return np.dot(v1_norm, v2_norm)


def preprocessing(input_text, del_stopwords=True, del_digit=True):
    # russian_stopwords = set(stopwords.words('russian'))
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


def get_w2v_vectors(lemmas_arr, file, text):
    feature_vec = np.zeros((300,), dtype='float32')
    for lemma in lemmas_arr:
        try:
            feature_vec = np.add(feature_vec, model.wv[lemma])
        except KeyError:
            feature_vec = np.add(feature_vec, np.zeros((300,), dtype='float32'))
    text = text.strip()
    text = text.replace('\xa0', '')
    text = text.replace('\n', '')
    dict_ = {text: feature_vec}
    return dict_


def get_w2v_vectors_query(lemmas_arr, text):
    """Получает вектор документа"""
    feature_vec = np.zeros((300,), dtype='float32')
    for lemma in lemmas_arr:
        try:
            feature_vec = np.add(feature_vec, model.wv[lemma])
        except KeyError:
            print(lemma)
            feature_vec = np.add(feature_vec, np.zeros((300,), dtype='float32'))
    text = text.strip()
    text = text.replace('\xa0', '')
    # text = text.replace('\n', '')
    dict_ = {text: feature_vec}
    return dict_


def save_w2v_base(dict_, base_dict):
    """Индексирует всю базу для поиска через word2vec"""
    base_dict.update(dict_)
    return base_dict


def pre_query():
    base_dict = {}
    path = 'D://avito/*.txt'  # note C:
    files = glob.glob(path)
    for name in files:
        with open(name, encoding='utf-8') as file:
            f = file.read()
            prepr = preprocessing(f, del_stopwords=True, del_digit=True)
            print('1')
            get_wv = get_w2v_vectors(prepr, file, f)
            print('2')
            save_wv = save_w2v_base(get_wv, base_dict)
    return save_wv


def w2v_search(query, save_wv):
    scores = []
    print('question - ', query)
    prepr = preprocessing(query, del_stopwords=True, del_digit=True)
    print('3')
    get_wv = get_w2v_vectors_query(prepr, query)
    for key, value in save_wv.items():
        for key1, value1 in get_wv.items():
            try:
                sim = similarity(value, value1)
                scores.append((key, sim))
            except ValueError:
                print('string')
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores[:5]
