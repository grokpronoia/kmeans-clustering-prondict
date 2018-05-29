import numpy as np
import pandas as pd
import math

from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import Pipeline

%config IPCompleter.greedy=True
%matplotlib inline
np.set_printoptions(precision=5, suppress=True)

location = '/Users/marrowgari/tensorflow/pron.txt'
data = pd.read_csv(location, names=('Words', 'Phones'))
words = data['Words']
phones = data['Phones']

print("Data shape --->", data.shape)
print("Words shape --->", words.shape)
print("Phones shape --->", phones.shape)
print('----------')
print("Data Sample:","\n", data[:6])

vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
X = vectorizer.fit_transform(phones)
print(X.shape)

true_k = 100
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=5000, n_init=1)
model.fit(X)

# combine vectorizer and cluster model into one object
phone_cluster_model = Pipeline(steps=[('encode', vectorizer), ('model', model)])

word_idx_lookup = {w: idx for idx, w in enumerate(words)}

def cluster_for_word(word):
    """
    For a given word, lookup phones and pass to clustering model, return cluster
    """
    word_idx = word_idx_lookup[word]
    phones_for_word = phones[word_idx]
    cluster = phone_cluster_model.predict([phones_for_word])[0]
    return cluster

def group_words_by_cluster(words):
    """
    Create groups of words by cluster (from phones)
    """
    words_by_cluster = {}
    for word in words:
        cluster = cluster_for_word(word)
        word_set = words_by_cluster.get(cluster, set())  # get the set of words, or empty set if not exists
        word_set.add(word)

        words_by_cluster[cluster] = word_set # save the modified set back into the grouping dict
    return words_by_cluster

words_by_cluster = group_words_by_cluster(words)

# for each cluster, print the cluster and 10 words in that cluster in no particular order
for cluster, cluster_words in words_by_cluster.items():
    cluster_words = list(cluster_words)
    words_in_cluster = len(cluster_words)
    print("Cluster[{} len{}]:{}".format(cluster, words_in_cluster, cluster_words[:10]))
    
