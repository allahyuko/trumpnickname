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





'''
pynlpir.open()
pynlpir.nlpir.AddUserWord(u"特朗普")
pynlpir.nlpir.AddUserWord(u"唐纳德")
pynlpir.nlpir.AddUserWord(u"唐纳德特朗普")
pynlpir.nlpir.AddUserWord(u"唐纳德约翰特朗普")
pynlpir.nlpir.AddUserWord(u"唐纳德川普")
pynlpir.nlpir.AddUserWord(u"川普")
pynlpir.nlpir.AddUserWord(u"希拉里克林顿")
pynlpir.nlpir.AddUserWord(u"希拉里")
pynlpir.nlpir.AddUserWord(u"希拉蕊")
pynlpir.nlpir.AddUserWord(u"科米詹姆斯")
pynlpir.nlpir.AddUserWord(u"科米")
pynlpir.nlpir.AddUserWord(u"伯尼桑德斯")
pynlpir.nlpir.AddUserWord(u"伯尼")
pynlpir.nlpir.AddUserWord(u"桑德斯")
pynlpir.nlpir.AddUserWord(u"巴拉克奥巴马")
pynlpir.nlpir.AddUserWord(u"奥巴马")
pynlpir.nlpir.AddUserWord(u"欧巴马")
pynlpir.nlpir.AddUserWord(u"伊万卡特朗普")
pynlpir.nlpir.AddUserWord(u"伊万卡")
pynlpir.nlpir.AddUserWord(u"川粉")
pynlpir.nlpir.AddUserWord(u"川黑")
pynlpir.nlpir.AddUserWord(u"希黑")
pynlpir.nlpir.AddUserWord(u"希婆")
pynlpir.nlpir.AddUserWord(u"床破")
pynlpir.nlpir.AddUserWord(u"比尔克林顿")
pynlpir.nlpir.AddUserWord(u"男克林顿")
pynlpir.nlpir.AddUserWord(u"男克")
pynlpir.nlpir.AddUserWord(u"女克")
#连接数据库，读取数据
con=sqlite3.connect("zhihu_election_answers.db")
sql="SELECT * from answers"
df1=pd.read_sql(sql,con,index_col='id')


#去重，如果文章标题，发布者和发布时间一样，则认为是重复文章
unique=df1['article_title']+df1['article_date']+df1['article_acount']
df1['unique']=unique
print(len(df1))
df2 = df1.drop_duplicates(['unique'])
print(len(df2))
#去除被发布者删除的文章，其阅读数在采集时丢失，赋值为-1
df=df2[df2.read_number!=-1]
print(len(df))


#提取10w+文章
df_10wan=df[df.read_number>=100000]
df_no10wan=df[df.read_number<100001]


file1= codecs.open('corpus.txt','w',"utf-8")
i=1

for item in df.content:
    item=re.sub(u"·",r"",item)
    filtrate = re.compile(u'[^\u4E00-\u9FA5\u0041-\u005a\u0061-\u007a]')
    item=filtrate.sub(r" ",item)
    item = Converter('zh-hans').convert(item)
    item = item.encode('utf-8')  
    seg_result=pynlpir.segment(item,pos_tagging=False)
    s=(" ".join(seg_result))
    #print i
    #i=i+1
    #print s
    file1.write(s+"\n")

file1.close()
print i
'''
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
