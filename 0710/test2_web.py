# -*- coding: utf-8 -*-
"""
Pythonで人工知能のWebサービスを実装する方法
http://aiweeklynews.com/archives/48462559.html

Created on Tue Jul 10 11:46:38 2018

@author: Akitaka
"""

#coding: UTF-8
# Here your code !
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pandas as pd
from sklearn.externals import joblib

#モデルの復元
SVM = joblib.load('neet.pkl')

# Flaskインスタンスを app という名前で生成する
app = Flask(__name__)

class HelloForm(Form):
    sayhello = TextAreaField('',[validators.DataRequired()])


# Webアプリケーション用のルーティングを記述
# index にアクセスした際の処理
@app.route('/')
def index():
    form = HelloForm(request.form)
    return render_template('neet_app.html', form=form)

@app.route('/hello', methods=['POST'])
def hello():
    form = HelloForm(request.form)
    if request.method == 'POST' and form.validate():
        name = request.form['neet']   
        #nameから顧客情報を引っ張ってきて、その顧客情報をモデルに入れて、NEETかどうか予測する
        df = pd.read_csv('neet2.csv')
        df_only = df[df['key'] == name]
        try:
            data_test = df_only.iloc[0:1,0:5].values
            label_prediction = SVM.predict(data_test)
            if label_prediction == 1:
                return render_template('neetHello.html', name=", You are NEET!")
            else:
                return render_template('neetHello', name=", You are Company farmer!")
        except:
            return render_template('neetHello',  name=", Your name is none (ToT)")
    return render_template('neet_app.html', form=form)

# 開発用サーバーを実行
if __name__ == '__main__':
    app.run(debug=True)