import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
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
with open('./conf/config.json') as f:
	config = json.load(f)

# config variables
data_path = config["data_path"]
random_seed = config["seed"]
model = config["model"]
model_path = config["model_path"]
test_size = config["test_size"]

doc=[]
classes=[]
# filename=[]
# filetype=[]
# magic_type=[]
if not os.path.exists(model_path):
    os.mkdir(model_path)

for root, directories, filenames in os.walk(data_path):
     for file in filenames:
            if file != ".DS_Store":
                file_name, file_extension = os.path.splitext(file)
                classes.append(root.split("/")[-1])
                # filename.append("{}/{}".format(root,file))
                # filetype.append(file_extension)
                # magic_type.append(magic.from_file("{}/{}".format(root,file), mime=True))
                file1 = open("{}/{}".format(root,file),"r",encoding='utf-8', errors='ignore')
                doc.append(file1.read())

df=pd.DataFrame()
df['doc']=doc
df['classes']=classes
# df['file']=filename
# df['type']=filetype


y=pd.get_dummies(df['classes'])
label_dict={}
for i in range(y.shape[1]):
    label_dict[i]=y.columns[i]
print("[INFO] saving label dictionary...")
joblib.dump(label_dict, "{}{}_label.pkl".format(model_path,model))
y =np.array(y)


stemmer = SnowballStemmer('english').stem
def stem_tokenize(text):
     return [stemmer(i) for i in word_tokenize(text)]

vectorizer = CountVectorizer(analyzer='word',lowercase=True,tokenizer=stem_tokenize)
X = vectorizer.fit_transform(df.doc.values)
joblib.dump(vectorizer, '{}{}_vectorizer.pkl'.format(model_path,model))

clf = MultinomialNB()
# mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
# mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])

y = np.argmax(y, axis=1)

# all_models = {"mult_nb":mult_nb,"mult_nb_tfidf":mult_nb_tfidf}

# clf=all_models.get(model)

clf.fit(X, y)

#print(cross_val_score(MultinomialNB(), X, y, cv=5).mean())

# dump classifier to file
print("[INFO] saving model...")
joblib.dump(clf, "{}{}_clf.pkl".format(model_path,model))
