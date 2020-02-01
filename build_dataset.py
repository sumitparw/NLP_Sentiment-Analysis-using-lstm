import nltk
import random
import pandas as pd
from nltk.tokenize import word_tokenize
import re
import ssl

from nltk.corpus import stopwords

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#nltk.download()

all_words = []
documents = []


def assign_label(x):
    if x[2] < 3.0:
        return "negative"
    elif x[2] > 3.0:
        return "positive"
    else:
        return "neutral"


def find_features(document, word_features):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


def return_train_test_data(file_path):
    df = pd.read_csv(file_path, header=None)
    df = df[df.columns[2:4]]
    df[2] = df.apply(assign_label, axis=1)
    inx = df[df[2] == 'neutral'].index
    df.drop(inx, inplace=True)
    df[2] = df[2].map({'negative': 0, 'positive': 1})
    negative = df[df[2] == 0][3].values
    positive = df[df[2] == 1][3].values
    all_words = []
    documents = []
    #stop_words = list(set(nltk.download('stopwords').w('english')))
    stop_words = list(set(stopwords.words('english')))
    allowed_word_types = ["J"]

    for p in positive:
        documents.append((p, "pos"))
        cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)
        tokenized = word_tokenize(cleaned)
        stopped = [w for w in tokenized if not w in stop_words]
        pos = nltk.pos_tag(stopped)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    for p in negative:
        documents.append((p, "neg"))
        cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)
        tokenized = word_tokenize(cleaned)
        stopped = [w for w in tokenized if not w in stop_words]

        neg = nltk.pos_tag(stopped)

        for w in neg:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:5000]
    featuresets = [(find_features(rev, word_features), category) for (rev, category) in documents]
    random.shuffle(featuresets)
    training_set = featuresets[:int(0.8 * (len(featuresets)))]
    testing_set = featuresets[int(0.8 * (len(featuresets))):]

    return training_set, testing_set
