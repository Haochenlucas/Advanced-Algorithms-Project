import nltk
from nltk.corpus import wordnet as wn
import numpy as np 

def add_to_workinglist(word, allWords, updateWorkingList):
    if(word not in allWords):
        updateWorkingList.add(word)

allWords = set()
adjacency_list = dict()
workingWords = {"science"}

for x in range(0, 4):
    updateWorkingList = set()
    #w_words[x] = workingWords
    for word in workingWords:
        allWords.add(word)
        adjacency_list[word] = set()
        # synonyms = set() 
        # antonyms = set() 
        # hypernyms = set()
        # homonyms = set()
        for syn in wn.synsets(word):
            hypernym = syn.hypernyms()
            hyponom = syn.hyponyms()
            for l in syn.lemmas():
                w = l.name()
                # synonyms.add(l.name())
                if w != word:
                    add_to_workinglist(w, allWords, updateWorkingList)
                    adjacency_list[word].add(w)
                if l.antonyms():
                    w = l.antonyms()[0].name()
                    # antonyms.add(l.antonyms()[0].name()) 
                    if w != word:
                        add_to_workinglist(l.antonyms()[0].name(), allWords, updateWorkingList)
                        adjacency_list[word].add(w)
            for i in hypernym:
                for j in i.lemmas():
                    w = j.name()
                    # hypernyms.add(j.name()) 
                    if w != word:
                        add_to_workinglist(j.name(), allWords, updateWorkingList)
                        adjacency_list[word].add(w)
            for i in hyponom:
                for j in i.lemmas():
                    w = j.name()
                    # homonyms.add(j.name()) 
                    if w != word:
                        add_to_workinglist(j.name(), allWords, updateWorkingList)
                        adjacency_list[word].add(w)
    workingWords = updateWorkingList
     
word_set = set()
for i in adjacency_list["science"]:
    for j in adjacency_list[i]:
        word_set.add(j)


final_word_set = set()   
p = 0
for s in word_set:
    if len(adjacency_list[s]) >5:
        p = p + 1
        final_word_set.add(s)

weight_mat = np.zeros(shape = (p,p))

word_list = list(final_word_set)
for i in range(p):
    for j in adjacency_list[word_list[i]]:
        if j in word_list:
           ind = word_list.index(j)
           weight_mat[i,ind] = 1 
           weight_mat[ind,i] = 1


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)  

if 0 in np.unique(np.sum(weight_mat,axis=1)):
    index = np.where(np.sum(weight_mat,axis=1)==0)

weight_mat_1 = np.delete(weight_mat,index,axis=0)
weight_mat_1 = np.delete(weight_mat_1,index,axis=1)

np.savetxt('wordNet.out', weight_mat_1)