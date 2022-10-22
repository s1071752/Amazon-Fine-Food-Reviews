#!/usr/bin/env python
# coding: utf-8

# ## 主題: Amazon Fine Food Reviews
# ## 分類器: 隨機森林

# ## About Dataset
# 
# This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.      
#   
# 
#     
# ## HW2
# >- 本次作業為情緒分析，資料集為Amazon Fine Food Reviews 的Reviews.csv
# >- https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
# 
# >- HW2 Kaggle 競賽網址：
# >- https://www.kaggle.com/t/3d90c24a5d754706833a49b7842739a9
#  
#   

# ## 1. 資料前處理
# 

# ### 1-1 讀取資料
# - 讀取csv檔前 10000 筆資料
# - 僅保留Text、Score兩個欄位

# In[31]:


import pandas as pd

# 主要使用 Reviews.csv這份資料
data = pd.read_csv("Reviews.csv", header=0, usecols=["Text","Score"], encoding='utf-8')[:10000]  #train
testData = pd.read_csv("test.csv", header=0, usecols=["Text"], encoding='utf-8')         # test

data.tail()


# In[32]:


print("Text欄位空值:", data['Text'].isnull().sum())
print("Score欄位空值:", data['Score'].isnull().sum())


# ### 1-2 資料轉換
# - 將Score欄位內值大於等於4的轉成1(positive), 其餘轉成0 (negative)
# 
#     
# 

# In[33]:


data.loc[data['Score']<4,'Score']=0
data.loc[data['Score']>=4,'Score']=1
data.head()


# ### 1-3 文字前處理
# 
# 1. 去除標點符號 
# 2. 統一大小寫
# 3. sentence segmentation (斷句)
# 4. word segmentation (斷詞)
# 5. stopword (去除停用詞，例如i, with, and)
# 6. pos 詞性標記
# 7. Lemmatization辭型還原(避免將同樣的字詞例如love/loves/loved視為不同的輸入)
# 
# 
# 參考文章: 
# https://clay-atlas.com/blog/2019/07/30/nlp-python-cn-nltk-kit/
# 

# In[34]:


def pos(tokens):
    pos = [nltk.pos_tag(token) for token in tokens]  # 詞性標記 pos
    wordnet_pos = []
    
    for p in pos:
        for word, tag in p:
            if tag.startswith('J'):
                wordnet_pos.append(nltk.corpus.wordnet.ADJ)
            elif tag.startswith('V'):
                wordnet_pos.append(nltk.corpus.wordnet.VERB)
            elif tag.startswith('N'):
                wordnet_pos.append(nltk.corpus.wordnet.NOUN)
            elif tag.startswith('R'):
                wordnet_pos.append(nltk.corpus.wordnet.ADV)
            else:
                wordnet_pos.append(nltk.corpus.wordnet.NOUN)

# 辭型還原 Lemmatization
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(p[n][0], pos=wordnet_pos[n]) for p in pos for n in range(len(p))]

    return tokens


# In[66]:


import nltk  #使用Nature Language Tool Kit (NLTK)進行文本處理
import string

####文本清理####
def text_preprocessing(method, text):
       
    text = text.translate(str.maketrans('', '', string.punctuation))  #刪去標點符號
    text = text.lower() # 統一轉為小寫

    sentences = nltk.sent_tokenize(text) # 斷句 
    tokens = [nltk.tokenize.word_tokenize(sent) for sent in sentences]  # 斷詞
    
    nltk_stopwords = nltk.corpus.stopwords.words("english")
    tokens = [token for token in tokens[0] if token not in nltk_stopwords] # 僅保留非停用字(去除停用字)
    
    tokens_pos = pos([tokens])  #詞性標記
    
    if method =='tfidf': 
        text=""        
        for t in tokens_pos:
            text += t
            text += " "
        return text  #需要回傳string
   
    elif method =='word2vec':
        
        return tokens_pos # 需要回傳list


# ### 1-4 文字轉向量
# 訓練模型前要將文字轉為機器可閱讀的形式，以前使用的one-hot/dummy會造成高維稀疏的向量矩陣，因此採計算詞向量的方法，以下實作 tf-idf 及 word2vec 並進行比較
# 
# #### (註: 1-4部分選擇其中一種執行即可，都執行只會有後寫入的word2vec結果)
# 

# ## Tfidf Vectorizer
# >- max_features 挑選出多少個有代表性的文字
# >- min_df/max_df 向量值高/低於此才會挑選，避免出現無代表性的詞彙
# >- 教學- https://ithelp.ithome.com.tw/articles/10228481
# 
# 變數 | 意義     
# ----------------- | -----------------  
# text_cleaned| 去除標點的train data      
# test_text_cleaned| 去除標點的test data    
# mix_text|  以上兩者相加        
# 

# #### 1-4-1 文本前處理

# In[67]:


text_cleaned = []  # 清乾淨的train data
test_text_cleaned = []  # 清乾淨的test data


for text in data['Text']:
    
    t = text_preprocessing('tfidf',text)
    text_cleaned.append(t)
   
for text in testData['Text']:
    
    t = text_preprocessing('tfidf', text)
    test_text_cleaned.append(t)
 


# In[68]:


print("清理前: ", data['Text'][10])
print("清理後: ", text_cleaned[10])
                         #    i

    
mix_text = [] #將文字合併，後續一起計算向量
mix_text.extend(text_cleaned)
mix_text.extend(test_text_cleaned) 


# #### 1-4-2 文字轉向量

# In[69]:


#                                                                                       【tf-idf 】
from sklearn.feature_extraction.text import TfidfVectorizer
import csv 


def toVec(textArr): 
                                                                                      
    vectorizer = TfidfVectorizer(stop_words='english', token_pattern="(?u)\\b\\w+\\b", max_features=500, min_df=0.0001, max_df=0.8)
    tfidf_X = (vectorizer.fit_transform(textArr)) #text to vector
    r = pd.DataFrame(tfidf_X.toarray(),columns=vectorizer.get_feature_names()) # show table
     
    return r  #vectors

    


# In[70]:


#如果是train, test分開算向量，最後選出的特徵不一致
# 因此使用以下的方式，一併作訓練
text_vector = toVec(mix_text)

#將剛剛轉好的文字vector 繪製出table-->出現max_features個新欄位 (max_features=500)
text_vector.tail()  


# #### 1-4-3 新資訊併入原本的數據集

# In[71]:


train_vector = text_vector[0:10000]
data_ok = pd.concat([data, train_vector], axis=1, join='inner') 

test_vector = text_vector[10000:15000]
test_vector.index = range(len(test_vector)) #要調整id才能正確join
testData_ok = pd.concat([testData, test_vector], axis=1, join_axes=[testData.index])

data_ok.head()


# *使用TfidfVectorizer轉為向量完成*

# ## Word2vec
# >- size：特徵向量的維度，預設值為100。
# >- min_count: 在n篇文章之中，出現在少於min_count篇的單字會被丟掉，預設值是5。
# >- max_count: 出現頻率大於max_count的不納入，避免納入沒有辨識性的單字。
# >- 教學- https://blog.csdn.net/weixin_45599022/article/details/109008368
# 
# 變數 | 意義     
# ----------------- | -----------------  
# train_cleaned| 去除標點的train data      
# test_cleaned| 去除標點的test data    
#      
# 

# #### 1-4-1文字前處理

# In[74]:



train_cleaned = []  # 清乾淨的train data
test_cleaned = []  # 清乾淨的test data
vocab = []  # 給模型的資料集

for text in data['Text']:
    
    t = text_preprocessing('word2vec',text)
    train_cleaned.append(t) 
    vocab.extend(t) # 只加入list內的元素而非整個[list]
   
for text in testData['Text']:
    
    t = text_preprocessing('word2vec', text)
    test_cleaned.append(t)
 


# In[49]:


print("清理前: ", data['Text'][10])
print("清理後: ", train_cleaned[10])     


# #### 1-4-2 文字轉向量
# word2vec: https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/
# 
# 教學: https://www.kaggle.com/code/jerrykuo7727/word2vec

# In[75]:


def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32") #生成都是0.的向量矩陣featureVec
    nwords = 0.

    # Index2word中包含了詞表中的所有詞，为了檢索速度，保存到set中
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.                               
            featureVec = np.add(featureVec, model.wv[word]) # 如果評論中的詞有出現在詞表中,用model.wv[]把字典中那個字的詞向量提取出來，加到featureVec
        
    # 將featureVec取平均
    if nwords != 0.:
        featureVec = np.divide(featureVec, nwords)
    return featureVec


# In[12]:


# model.wv['like']  
# 若上面那段code的num_features=5  --> featureVec長度為5，一開始內容都是0.--> [0.0.0.0.0.]
# 若句子中有like單字，like的向量為[-1.625,  4.104, -1.567,  2.106, -6.5126]，featureVec會各自加上這5個向量
# 重複此步驟直到句子中單字都檢索完畢



# In[76]:


from gensim.models import Word2Vec
from gensim.test.utils import common_texts  #使用common_texts訓練用的詞彙

model = Word2Vec([vocab], min_count=1, vector_size=50) # vector_size詞向量的維度大小(預設100)
#model.build_vocab(vocab)  # prepare the model vocabulary
model.train([vocab], total_examples=model.corpus_count, epochs=model.epochs)  # train word vectors

vec = []

def word2vec(text):
    for sentence in text:
        vec.append(makeFeatureVec(sentence, model, 50))  # 對每一份評論中的所有詞向量取平均      
         # num_features 要跟word2vec model的參數設定相同(50)
        
    return vec


# #### 1-4-3 向量併入原本的數據集(或是應該取平均試試?)

# In[77]:


import numpy as np

train_vector = pd.DataFrame(word2vec(train_cleaned))
data_ok = pd.concat([data, train_vector], axis=1, join='inner') 

test_vector =  pd.DataFrame(word2vec(test_cleaned))
test_vector.index = range(len(test_vector)) #要調整id才能正確join
testData_ok = pd.concat([testData, test_vector], axis=1, join='inner')

data_ok.head()
# 因為num_features設為50，故增加50列


# *使用Word2vec轉為向量完成*

# ## 2. 建立模型( Random forest )

# >- 輸入的資料型態為 pd.DataFrame(文字的部分要轉為向量)
# >- n_estimators : 隨機森林中的樹數量，我覺得設定高一些效果較好，相對地挑選代表詞彙時要挑選更多個詞。  
# >- min_samples_split : 劃分出新分支點所需要的最小樣本數
# 
# 

# In[78]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

                                              ####  【輸出預測結果，上傳至kaggle】  ####

def predict_testData(k, model):
    
    predicted = model.predict(testData_ok.drop(labels=['Text'], axis=1))
# 讀取test.csv，輸出結果至test.csv
    id = [i for i in range(1, 5000+1)]
    results = {
            'ID': id,   #輸出共有兩行，第一行是角色編號，第二行是(預測結果)是否死亡
            'Score': predicted
    }

    submission = pd.DataFrame(results)

    submission.to_csv(str(k)+"-submission.csv", index=False, header=1)

    
    
    
                                                ####  【進行k-fold cross-validation】  #### 

def k_fold(k, data):  #.copy()
    
    folds = np.array_split(data, k) #將data切成k份，其中1份當測試集，剩餘k-1份當訓練集建立模型
    
    accuracy = 0.0
    
    for i in range(k): # cross validation

        train_data = folds.copy()  
        del train_data[i]       
        train_data = pd.concat(train_data, sort=False)
        
        test_data = folds[i]

        x_train = train_data.drop(labels=['Score','Text'], axis=1)  #id--> axis 0   attribute-->axis 1 
        y_train = train_data['Score']
        x_test = test_data.drop(labels=['Score','Text'], axis=1) # testdata
        y_test = test_data['Score']                              # testdata label

        clf = RandomForestClassifier(n_estimators=300, min_samples_split=2)
        clf.fit(x_train, y_train)
        
        accuracy += clf.score(x_test, y_test)
    
        predict_testData(k, clf)
        
    return accuracy/k  

#輪流將k份的每份資料都當 測試集，其餘當訓練集建立模型，因此會進行k次，k次都計算出Accuracy
#將k次的Accuracy平均即為output


# ## 3. tf-idf 、 word2vec 之結果比較
# 
# | 方法 | 計算方式 | 結果比較     
# | :----: | :--- | :--- 
# | tf-idf | 根據詞彙在整份文章的出現頻率以及在其它文本的出現頻率，判斷詞彙的重要性將文字轉換成向量 | 可以計算字詞在主題上的代表性，像是在此範例中有提取出glad, prefer, yummy等詞，我覺得對此次測試資料的預測較合適    
# | word2vec | 將文中詞彙計算cosin值，cosin值相近的詞彙意義較接近 | 比較不參考文字在文本中的重要性，若將詞彙向量取平均作為文本的平均向量在此次預測的效果會不好，沒有明顯看出正向評價與負向評價的向量值差異，應該針對其中的關鍵詞作加權    
#         
# 

# ###  tf-idf 

# In[73]:


k_fold(4, data_ok) #### (註: 1-4部分選擇其中一種執行即可，都執行只會有後寫入的word2vec結果)


# ### word2vec

# In[79]:


k_fold(4, data_ok)### (註: 1-4部分選擇其中一種執行即可，都執行只會有後寫入的word2vec結果)


# In[ ]:




