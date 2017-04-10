#encoding=utf-8

#import jieba
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import re
import codecs
from gensim.models import word2vec
import cython
#jieba.load_userdict('user_dictionary.txt')
import pynlpir
from langconv import *
from PIL import Image, ImageDraw, ImageFont
from scipy.misc import imread
import os
from os import path
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8') 

stopwordfile=codecs.open("stopwords.txt","r","utf-8")
stopwords=stopwordfile.readlines()
stopwords=[re.sub("\r\n|\n","",word) for word in stopwords]

file1= codecs.open('corpus_no_stopword1.txt','r',"utf-8")
corpus=file1.readlines()
corpus1=file1.read()
result={}
i=0
for text in corpus:
    for word in text.lower().split():
        i=i+len(word)
        if (len(word)>=2 and (word not in stopwords)):
            if word not in result:
                result[word]=0
            result[word]+=1

print i#计算总单词数

result[u"特朗普"]
result[u"川普"]
frequency= sorted(result.iteritems(), key=lambda d:d[1], reverse = True)


d = path.dirname('.')
color_mask = imread("05a9Db.jpg")

from wordcloud import WordCloud
cloud = WordCloud(
        #设置字体，不指定就会出现乱码
        font_path="simhei.ttf",
        #font_path=path.join(d,'simsun.ttc'),
        #设置背景色
        background_color='black',
        #词云形状
        mask=color_mask,
        #允许最大词汇
        max_words=300,
        #最大号字体
        max_font_size=200
    )
word_cloud = cloud.generate_from_frequencies(result)
word_cloud.to_file("trump.jpg")



sentences = word2vec.LineSentence("corpus_no_stopword1.txt")

model1 = word2vec.Word2Vec(sentences, size=200,window=5,min_count=10) 
model2 = word2vec.Word2Vec(sentences, size=200,window=10,min_count=10)
model3 = word2vec.Word2Vec(sentences, size=200,window=15,min_count=10) 

y_t_1 = model1.most_similar(u"特朗普", topn=20)
y_t_2 = model2.most_similar(u"特朗普", topn=20)
y_t_3 = model3.most_similar(u"特朗普", topn=20)

y_h_1 = model1.most_similar(u"希拉里", topn=20)
y_h_2 = model2.most_similar(u"希拉里", topn=20)
y_h_3 = model3.most_similar(u"希拉里", topn=20)


y_o_1 = model1.most_similar(u"奥巴马", topn=20)
y_o_2 = model2.most_similar(u"奥巴马", topn=20)
y_o_3 = model3.most_similar(u"奥巴马", topn=20)

#输出结果
for item in y_t_1:
    print item[0],item[1]



relation1 = model1.most_similar(positive=[u"希拉里",u"共和党"],negative=[u"特朗普"], topn=10)
for item in relation1:
    print item[0], item[1]

relation2 = model1.most_similar(positive=[u"川普",u"加州"],negative=[u"希拉里"], topn=10)
for item in relation2:
    print item[0], item[1]
