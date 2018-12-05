# Advanced-Algorithms-Project


## Overview
This is a prgram 


## Datasets

* WordNet: 
Form a unweighted graph starting with the word "science". A vertex represent a word and an edge stands for the realationship between the two words. Add a node and a edege if the following relationship exist between two words: synonyms, antonyms, hypernym, hyponym. We only include the nodes that is within 3 edges from the starting word "science".

* NCAA: 
Using NCAA football matches in 2013 to form an unweighted graph. Each team is a vertex and link the two with an edge if they have a match against each other in the season.


## Simulations
We use stochatic block model to form a total of 27 different graphs using the following settings: 
* total number of vertices in the graph: [100,250,500]
* total number of clusters in the graph: [5,7,10]
* the probability of an edge connecting two vertices in the same cluster: [0.4, 0.7, 0.9] 


## How to run
Running pickQ.py will use the two datasets (WordNet and NCAA) to form a 2D-plot showing the clusters. It will also draw a 3D-plot with x,y,z being number of eigen_vector, number of cluster and Modularity.

Furthermore, it will run 27 simulations and write the data needed into files. Runing simulation_plot.py will draw two more plot: 
* relationship between top k - 1 eigenvalue / top k eigenvalue' \n and top k eigenvalue/ top k + 1 eigenvalue 
* ratio between best number of eigen_vector \nand best number of cluster

## Requirement

* stochatic block model (included)
* Python
* scipy
* sklearn
* numpy
* matplotlib
* networkx
* pickle
* nltk
