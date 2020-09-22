# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:41:15 2020

@author: delli
"""

import numpy as np
from sklearn.cluster import KMeans
import time 
import os

def recursive(features, idx_features, b, depth, n, numbers_of_objects):
    
    data = features[idx_features]
    dList = []
    children = []
    kmeans = None
    obj_array = None
    if data.shape[0] > b and depth > 1:
            kmeans = KMeans(n_clusters = b, random_state = 0).fit(data)
            for i in range(b):
                
                idx_b = [idx_features[l] for l, ll in enumerate(kmeans.labels_) if ll == i]
                new_dList = recursive(features, idx_b, b, depth-1, n+len(dList)+1,numbers_of_objects)
                
                dList += new_dList
                children.append(new_dList[-1]['i'])
    else:
        obj_array= np.zeros(numbers_of_objects)
    dList.append({'i': n, 'sift': idx_features,'model': kmeans, 'children': children,'objects':obj_array })
    return dList
    
    #for child in dList[0]['children']:
    #    recursive(dList, depth)

def hi_kmeans(data, b, depth, number_of_objects):
    dList = []
    
    dList=recursive(data,np.arange(data.shape[0]), b, depth,0,number_of_objects)
    dList = sorted(dList, key=lambda k:k['i']) 
        
    return dList



def recursiveExploration(Tree, feature, node):   
    if(Tree[node]['model']!=None):
        cluster=Tree[node]['model'].predict(feature)
        node = recursiveExploration(Tree, feature, Tree[node]['children'][cluster[0]])
        
    return node
    
#Data has to be (obj x n_features x 128)
def link_word_to_object(Tree,data):
    for i,obj in enumerate(data):
        for feature in obj:
            leaf =recursiveExploration(dList, feature.reshape(1,-1), 0)
            Tree[leaf]['objects'][i]+=1
        


def ReadData(path):
    data = []
    for file in os.listdir(path):
        tmp=np.load(path+file, allow_pickle=True)
        #print(file,tmp.shape)
        tmp = tmp.reshape(-1,tmp.shape[2])
        data.append(tmp)
    return np.array(data)


#sift_list = np.random.randint(0,500,(50,300*3, 128))
'''
data= ReadData("Data2/server/sift/")
number_of_objects=data.shape[0]
sift_list = np.concatenate( data, axis=0 )

t1 = time.time()
dList= hi_kmeans(sift_list,2,3,number_of_objects)
t2 = time.time()
print(t2-t1)
link_word_to_object(dList,data)
print(time.time()-t2)
'''
tree = np.load("Data2/testTree.npy",allow_pickle=True).tolist()
leafs = list(filter(lambda x: len(x['children'])==0,tree))
f = np.array(list(map(lambda x: x['objects'] , leafs)))
F = np.array(list(map(lambda x: x.shape[0], data) ))
K = np.array(list(map(lambda x: np.sum(x!=0), f) ))


W = f/F * np.log2(K/number_of_objects).reshape(-1,1)