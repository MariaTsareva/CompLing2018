from d2v_search import d2v_search, ready_for_query
from w2v_search import w2v_search, pre_query
from okapi import fin_ok, freq
import os
import operator
import glob
from flask import Flask, render_template, request, url_for, redirect
app = Flask(__name__)
rfq = ready_for_query()
pr_qr = pre_query()


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


texts, N, av_len = open_()
# q, av_len = freq(query, texts)


@app.route('/')
def search_():
    return render_template('search.html')


@app.route('/search', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        meth = request.form['search type']
        query = request.form['Query']
        if meth == 'Okapi BM25':
            scores = fin_ok(query, texts, N, av_len)
        elif meth == 'Word2Vec':
            scores = w2v_search(query, pr_qr)
        elif meth == 'Doc2Vec':
            scores = d2v_search(query, rfq)
    return render_template("query_res.html", scores=scores)


if __name__ == '__main__':
    app.run(host='localhost', port=5003,debug = True)