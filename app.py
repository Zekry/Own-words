import flask
import os
import urllib.request
from flask import flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from gensim.summarization import keywords
from nltk.corpus import wordnet as wn
import requests
import time
import pandas as pd
import gensim.downloader as api
import pickle
import numpy as np

app = flask.Flask(__name__)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = 'uploads//'
app.config["file_upload"] = 'static//'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'doc', 'docx'])


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
	return render_template('upload.html')

# @app.route('/', methods=['POST'])
# def upload_file():
# 	if request.method == 'POST':
#         # check if the post request has the file part
# 		if 'file' not in request.files:
# 			flash('No file part')
# 			return redirect(request.url)
# 		file = request.files['file']
# 		if file.filename == '':
# 			flash('No file selected for uploading')
# 			return redirect(request.url)
# 		if file and allowed_file(file.filename):
# 			filename = secure_filename(file.filename)
# 			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
# 			flash('File successfully uploaded')
# 			return render_template('loading.html')
#         # else:
# 		# 	flash('Allowed file types are txt, pdf, doc, docx')
# 		# 	return redirect(request.url)

# text = "The Wandering Earth, described as China’s first big-budget science fiction thriller, quietly made it onto screens at AMC theaters in North America this weekend, and it shows a new side of Chinese filmmaking — one focused toward futuristic spectacles rather than China’s traditionally grand, massive historical epics. At the same time, The Wandering Earth feels like a throwback to a few familiar eras of American filmmaking. While the film’s cast, setting, and tone are all Chinese, longtime science fiction fans are going to see a lot on the screen that reminds them of other movies, for better or worse."

@app.route('/', methods=['POST'])
def process_text():

    if request.method == 'POST':
        result = flask.request.form
    return render_template('upload.html')
        kw = keywords(text, ratio=0.2, split=False, scores=False, pos_filter=None, deacc=True).split()
        lms = []
        for word in kw:
            lemma = wn.morphy(word)
            if lemma is None:
                lms.append(word)
            else:
                lms.append(lemma)
        model = pickle.load(open('vectors.pkl', 'rb'))
        related_words = {}
        for w in lms:
            try:
                words = [w, model.most_similar(w, topn=4)[0][0], model.most_similar(w, topn=4)[1][0],
                         model.most_similar(w, topn=4)[2][0], model.most_similar(w, topn=4)[3][0]]
                related_words.update({w: words})
            except:
                pass
        new = pd.DataFrame()
        for key in related_words:
            list_ = []
            for value in related_words[key]:
                try:
                    params = {'key':'dict.1.1.20190811T153301Z.6c88b4d06dce90ee.7ef3e4ec8ade64d1b8999ef0b5d1c6667df87c9f',
                              'lang': 'en-de', 'text':value}
                    response = requests.get("https://dictionary.yandex.net/api/v1/dicservice.json/lookup?", params=params)
                    hmm = tuple([response.json()['def'][0]['text'], response.json()['def'][0]['tr'][0]['text']])
                    list_.extend([hmm])
                except:
                    pass
                    list_.extend([None])
            new[key] = list_
        results = []
        for row in new.T.to_numpy():
            results.append([cell for cell in row if cell is not None])
            final_table = pd.DataFrame(results)[[0,1,2]]
        print(final_table)
        text_vocab = final_table[[0]].T
        related_vocab = final_table[[1,2]].T
        print(text_vocab)
        print(related_vocab)
        return render_template('result.html',tables=[text_vocab.to_html(classes='main'), related_vocab.to_html(classes='related')],
        titles = ['na', 'Vocabulary from text', 'Related Vocabulary'], text=text)
    return render_template('upload.html')

@app.route('/result')
def result_page():
    if request.method == 'POST':
        result = request.form
    text = result['text']
    return render_template('result.html', text=text)


if __name__ == "__main__":

    Host = '127.0.0.1'
    port = 2045
    app.run(Host, port, debug = True)
