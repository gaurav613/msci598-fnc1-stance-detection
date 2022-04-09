## MSCI 598 - Final Project ##
## Gaurav Mudbhatkal - 20747018 ##

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset

import os
import csv

from utils.config import stop_words
from utils.config import DATA_PATH
from utils.config import LABEL_MAP


class FNCDataset(Dataset):
    def __init__(self, mode='train'):
        super(FNCDataset, self).__init__()
        self.mode = mode
        if mode == 'train':
            data_stances_path = os.path.join(DATA_PATH, mode+'_stances.csv')
            data_bodies_path = os.path.join(DATA_PATH, mode+'_bodies.csv')
        else:
            data_stances_path = os.path.join(DATA_PATH, mode+'_stances_unlabeled.csv')
            data_bodies_path = os.path.join(DATA_PATH, mode+'_bodies.csv')

        stances = self.read(data_stances_path)
        articles = self.read(data_bodies_path)

        body = {}
        self.headline_body_pairs = []

        for article in articles:
            body[int(article['Body ID'])] = article['articleBody']

        for stance in stances:
            stance = dict(stance)
            headline_body_pair = {}
            if self.mode == 'train':
                headline_body_pair['text'] = (stance['Headline'], body[int(stance['Body ID'])])
            else:
                headline_body_pair['text'] = (stance['Headline'], body[int(stance['Body ID'])], int(stance['Body ID']))
            headline_body_pair['label'] = stance['Stance'] if self.mode == 'train' else 'to be predicted'
            self.headline_body_pairs.append(headline_body_pair)

        self.previous_pairs = self.headline_body_pairs

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.headline_body_pairs[index]['text'], self.headline_body_pairs[index]['label']
        else:
            return self.headline_body_pairs[index]['text'], self.headline_body_pairs[index]['label'],\
                   self.previous_pairs[index]['text'], self.previous_pairs[index]['label']

    def __len__(self):
        return len(self.headline_body_pairs)

    def read(self, filename):
        rows = []

        with open(filename, 'r', encoding='utf-8') as table:
            r = csv.DictReader(table)
            for line in r:
                rows.append(line)

        return rows

def pipeline_train(train, test, vector_size):
    """
    Method to prepare the preprocessing pipline using the training data - this method generates appropriate vectorizers - including bag-of-words, tf, and tf-idf 
    Args:
        train: FNCDataset object for training set
        test: FNCDataset object for testing set
        vector_size: size of vectors
    Returns:
        bow_vectorizer: bag-of-words vectors
        tfreq_vectorizer: only term-frequency vectors
        tfidf_vectorizer: tf-idf vectors
    """

    # Initialise
    heads = []
    heads_track = {}
    bodies = []
    bodies_track = {}
    id_ref = {}
    cos_track = {}
    test_heads = []
    test_heads_track = {}
    test_bodies = []
    test_bodies_track = {}
    head_tfidf_track = {}
    body_tfidf_track = {}

    # Identify unique headlines and bodies from their pairs
    for text, label in train:
        headline = text[0]
        body = text[1]
        if headline not in heads_track:
            heads.append(headline)
            heads_track[headline] = 1
        if body not in bodies_track:
            bodies.append(body)
            bodies_track[body] = 1

    for text, label, pre_text, pre_label in test:
        headline = text[0]
        body = text[1]
        if headline not in test_heads_track:
            test_heads.append(headline)
            test_heads_track[headline] = 1
        if body not in test_bodies_track:
            test_bodies.append(body)
            test_bodies_track[body] = 1

    # Create dictionary for unique headlines and bodies
    for i, elem in enumerate(heads + bodies):
        id_ref[elem] = i

    # Create vectorizers and BOW, TF, TF-IDF vectorizers from training set
    bow_vectorizer = CountVectorizer(max_features=vector_size, stop_words=stop_words)
    bow = bow_vectorizer.fit_transform(heads + bodies)  

    tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)
    tfreq = tfreq_vectorizer.transform(bow).toarray() 

    tfidf_vectorizer = TfidfVectorizer(max_features=vector_size, stop_words=stop_words). \
        fit(heads + bodies + test_heads + test_bodies)  

    new_pairs = []

    # concatenate tf vectors for headlines and bodies with the cosine similarity score
    for text, label in train:
        headline = text[0]
        body = text[1]
        head_tf = tfreq[id_ref[headline]].reshape(1, -1)
        body_tf = tfreq[id_ref[body]].reshape(1, -1)
        if headline not in head_tfidf_track:
            head_tfidf = tfidf_vectorizer.transform([headline]).toarray()
            head_tfidf_track[headline] = head_tfidf
        else:
            head_tfidf = head_tfidf_track[headline]
        if body not in body_tfidf_track:
            body_tfidf = tfidf_vectorizer.transform([body]).toarray()
            body_tfidf_track[body] = body_tfidf
        else:
            body_tfidf = body_tfidf_track[body]
        if (headline, body) not in cos_track:
            tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
            cos_track[(headline, body)] = tfidf_cos
        else:
            tfidf_cos = cos_track[(headline, body)]
        feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        pair = {}
        pair['text'] = torch.from_numpy(feat_vec)
        pair['label'] = LABEL_MAP[label]
        new_pairs.append(pair)

    train.headline_body_pairs = new_pairs

    return bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer


def pipeline_test(test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
    """
    Method to preprocess the testing set - this method generates appropriate vectors - tf vector with cosine similarity of tf vectors
        Args:
        test: FNCDataset object of testing set
        bow_vectorizer: bow vectorizer
        tfreq_vectorizer: term-frequency vectorizer
        tfidf_vectorizer: tf-idf vectorizer
    Returns:
        modifies the text from the test set to it's appropriate feature vectors
    """

    # Initialize tracking
    heads_track = {}
    bodies_track = {}
    cos_track = {}

    new_pairs = []

    # Vectorize test set
    for text, label, pre_text, pre_label in test:
        headline = text[0]
        body = text[1]
        if headline not in heads_track:
            head_bow = bow_vectorizer.transform([headline]).toarray()
            head_tf = tfreq_vectorizer.transform(head_bow).toarray()[0].reshape(1, -1)
            head_tfidf = tfidf_vectorizer.transform([headline]).toarray().reshape(1, -1)
            heads_track[headline] = (head_tf, head_tfidf)
        else:
            head_tf = heads_track[headline][0]
            head_tfidf = heads_track[headline][1]
        if body not in bodies_track:
            body_bow = bow_vectorizer.transform([body]).toarray()
            body_tf = tfreq_vectorizer.transform(body_bow).toarray()[0].reshape(1, -1)
            body_tfidf = tfidf_vectorizer.transform([body]).toarray().reshape(1, -1)
            bodies_track[body] = (body_tf, body_tfidf)
        else:
            body_tf = bodies_track[body][0]
            body_tfidf = bodies_track[body][1]
        if (headline, body) not in cos_track:
            tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
            cos_track[(headline, body)] = tfidf_cos
        else:
            tfidf_cos = cos_track[(headline, body)]
        feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        pair = {}
        pair['text'] = torch.from_numpy(feat_vec)
        pair['label'] = 'unknown'
        new_pairs.append(pair)

    test.previous_pairs = test.headline_body_pairs
    test.headline_body_pairs = new_pairs

