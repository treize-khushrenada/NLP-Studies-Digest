from flask import Flask, render_template, request, redirect, url_for
from joblib import load
import pandas as pd
from tweets import get_related_tweets


pipeline = load("text_classification.joblib")


def requestResults(name):
    tweets = get_related_tweets(name)

    tweets['prediction'] = pipeline.predict(tweets['tweet_text'])
    data = str(tweets.prediction.value_counts()) + '\n\n'
    tweets['hate speech'] = tweets['prediction'].map({1: 'Yes', 0: 'No'})
    tweets.drop(['prediction', 'tweet_id'], axis = 1, inplace=True)
    html = tweets.to_html()

    # tweets = tweets.drop(['prediction'], axis = 1, inplace=True)
    
    return html, data
    # return data + str(tweets)


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        return redirect(url_for('success', name=user))


@app.route('/success/<name>')
def success(name):
    html, data = requestResults(name)
  
    
    return html
    # return render_template('simple.html',  tables=[df.to_html(classes='data', header="true")])
    # return "<xmp>" + str(requestResults(name)) + " </xmp> "


if __name__ == '__main__':
    app.run(debug=True)