# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:41:15 2020

@author: delli
"""

import numpy as np
from sklearn.cluster import KMeans
import time

def recursive(features, idx_features, b, depth, n, n_objects):

    data = features[idx_features]
    dList = []
    children = []
    kmeans = None
    obj_array = None
    if data.shape[0] > b and depth > 1:
            kmeans = KMeans(n_clusters = b, random_state = 0).fit(data)
            for i in range(b):

                idx_b = [idx_features[l] for l, ll in enumerate(kmeans.labels_) if ll == i]
                new_dList = recursive(features, idx_b, b, depth-1, n+len(dList)+1, n_objects)

                dList += new_dList
                children.append(new_dList[-1]['i'])
    else:
        obj_array= np.zeros(n_objects)
    dList.append({'i': n, 'model': kmeans, 'children': children, 'objects': obj_array})
    return dList

def hi_kmeans(data, b, depth, n_objects):
    dList = []

    dList = recursive(data, np.arange(data.shape[0]), b, depth, 0, n_objects)
    dList = sorted(dList, key = lambda k: k['i'])

    return dList

def recursiveExploration(tree, feature, node):

    if tree[node]['model'] != None:

        cluster = tree[node]['model'].predict(feature)
        node = recursiveExploration(tree, feature, tree[node]['children'][cluster[0]])

    return node

# Data has to be (obj x n_features x 128)
def link_word_to_object(tree, data):

    for i,obj in enumerate(data):

        for feature in obj:

            leaf = recursiveExploration(tree, feature.reshape(1, -1), 0)
            tree[leaf]['objects'][i] += 1

def mapLeafsToNode(leafs):

    return np.array(list(map(lambda x: x['i'], leafs)))

def queryTree(tree, query, leafs):

    result = np.zeros((query.shape[0], len(leafs)))
    leafIdx = mapLeafsToNode(leafs)
    for i, obj in enumerate(query):
        for feature in obj:
            leaf = recursiveExploration(tree, feature.reshape(1, -1), 0)
            leaf = np.where(leafIdx == leaf)[0][0]
            result[i][leaf] +=1
    return result

def ReadData(path, n_features = 1e5, isClient = False):

    data = []
    for i in range(1, 50+1): # 50 is number of objects

        tmp = '_t' * isClient
        tmp = np.load(path + 'obj' + str(i) + tmp + '.npy', allow_pickle = True)[:, :n_features, :]
        # print(file, tmp.shape)
        tmp = tmp.reshape(-1, tmp.shape[2])
        data.append(tmp)
    return np.array(data)

def computeScore(q, d):

    s =[]
    for i,query in enumerate(q):

        s.append(np.linalg.norm(query/np.linalg.norm(query) - d/np.linalg.norm(d), axis = 1))
    return s

def createTree(data, n_features, b, depth):

    sift_list = np.concatenate(data, axis = 0)
    n_objects = data.shape[0]

    tree = hi_kmeans(sift_list, b = b, depth = depth, n_objects = n_objects)
    link_word_to_object(tree, data)
    leafs = list((filter(lambda x: len(x['children']) == 0, tree)))
    return tree, leafs

def computeTF_IDF(tree, leafs, data):

    n_objects = data.shape[0]

    f = np.array(list(map(lambda x: x['objects'], leafs)))
    F = np.array(list(map(lambda x: x.shape[0], data)))
    K = np.array(list(map(lambda x: np.sum(x != 0), f)))

    W = f/F * np.log2(K/n_objects).reshape(-1, 1)

    return W, f

def computeSortedScore(W, n, m, top = 1):

    q = n @ W
    d = (m.T @ W).T

    s = np.array(computeScore(q, d))

    sArgSorted = np.argsort(s, axis = 1)[:, :top]
    return sArgSorted

def top1(s):

    i = range(s.shape[0])
    return np.sum(s[:, 0] == i)

def topN(s, n = 1):

    return np.sum([np.any(i == ss) for i, ss in enumerate(s)])
    # tree = np.load('Data2/testtree.npy', allow_pickle = True).tolist()

timeList = []
top1List = []
top5List = []
param = [(4, 3), (4, 5), (5, 7)]
feat = np.arange(100, 1000+100, 100)
config =[(b, d, f) for (b, d) in param for f in feat]

for b, depth, n_features in config:

    t1 = time.time()
    data = ReadData('Data2/server/sift/', n_features = n_features, isClient = False)
    tree,leafs = createTree(data, n_features, b, depth)
    W, m = computeTF_IDF(tree, leafs, data)
    for r in [1, 0.9, 0.7, 0.5]:

        queryData = ReadData('Data2/client/sift/', n_features = int(n_features*r), isClient = True)
        n = queryTree(tree, queryData, leafs)

        s = computeSortedScore(W, n, m, 5)
        score1 = top1(s)
        score5 = top5(s)
        t = time.time() - t1

        timeList.append(t)
        top1List.append(score1)
        top5List.append(score5)

        print('Feature:', n_features, 'b:', b, 'd:', depth, 'r:', r, 't: {:.0f}'.format(t), 'top1:', score1, 'top5:', score5)
