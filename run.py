import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import pickle as cPickle
import os
import re
import collections
from tqdm import tqdm
import json

# load the user configs
with open('conf/config.json') as f:
	config = json.load(f)

# config variables
data_path = config["data_path"]
random_seed = config["seed"]
model = config["model"]
model_path = config["model_path"]
file_ = config["file"]
test = config["test_path"]
result = config["output_path"]

stemmer = SnowballStemmer('english').stem
def stem_tokenize(text):
     return [stemmer(i) for i in word_tokenize(text)]

clf = joblib.load("{}{}_clf.pkl".format(model_path,model))
label_dict= joblib.load("{}{}_label.pkl".format(model_path,model))
tfidf = joblib.load("{}{}_vectorizer.pkl".format(model_path,model))
doc=[]
filename = []
for root, directories, filenames in os.walk(test):
     for file in filenames:
            if file != ".DS_Store":
                file_name, file_extension = os.path.splitext(file)
                filename.append("{}/{}".format(root,file))
                # filetype.append(file_extension)
                # magic_type.append(magic.from_file("{}/{}".format(root,file), mime=True))
                file1 = open("{}/{}".format(root,file),"r",encoding='utf-8', errors='ignore')
                doc.append(file1.read())

df=pd.DataFrame()
df['doc']=doc
df['filename']=filename


X = tfidf.transform(df.doc.values)

pred = clf.predict(X)
output = pd.DataFrame()
output["file"]=df['filename']
output["pred"]=pred
output=output.replace({"pred": label_dict})
output.to_csv(result,index=False)
