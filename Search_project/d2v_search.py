import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import numpy as np
from gensim import matutils
import string
from pymorphy2 import MorphAnalyzer
morph = MorphAnalyzer()
import os
import operator
import glob
base = 'https://www.avito.ru/moskva/odezhda_obuv_aksessuary/'
model = Doc2Vec.load('D:/Doc2Vec_100s_1000e')


def similarity(v1, v2):
    v1_norm = matutils.unitvec(np.array(v1))
    v2_norm = matutils.unitvec(np.array(v2))
    return np.dot(v1_norm, v2_norm)


def preprocessing(input_text, del_digit=True):
    words = [x.lower().strip(string.punctuation+'»«–…') for x in word_tokenize(input_text)]
    lemmas = [morph.parse(x)[0].normal_form for x in words if x]
    lemmas_arr = []
    for lemma in lemmas:
        if del_digit:
            if lemma.isdigit():
                continue
        lemmas_arr.append(lemma)
#     lemmas_ = ' '.join(lemmas_arr)
    return lemmas_arr


def get_d2v_vectors(lemmas_arr, text):
    """Получает вектор документа"""
    new_vec = model.infer_vector(lemmas_arr)
    text = text.strip()
    text = text.replace('\xa0', '')
    # text = text.replace('\n', '')
    dict_ = {text :  new_vec}
    return dict_


def save_d2v_base(dict_, base_dict):
    """Индексирует всю базу для поиска через doc2vec"""
    base_dict.update(dict_)
    return base_dict


def ready_for_query():
    base_dict = {}
    path = 'D://avito/*.txt'  # note C:
    files = glob.glob(path)
    for name in files:
        with open(name, encoding='utf-8') as file:
            f = file.read()
            prepr = preprocessing(f, del_digit=True)
            print('1')
            get_dv = get_d2v_vectors(prepr, f)
            print('2')
            save_dv = save_d2v_base(get_dv, base_dict)
    return save_dv


def d2v_search(query, save_dv):
    scores = []
    print('question - ', query)
    prepr = preprocessing(query, del_digit=True)
    print('3')
    get_dv = get_d2v_vectors(prepr, query)
    for key, value in save_dv.items():
        for key1, value1 in get_dv.items():
            try:
                sim = similarity(value, value1)
                scores.append((key, sim))
            except ValueError:
                print('string')
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores[:5]
