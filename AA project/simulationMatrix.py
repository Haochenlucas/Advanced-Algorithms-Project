import os
import numpy as np 
from sbm.sbm import SBM
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans

os.chdir("./sbm")

class SBM_Matrix:
    def __init__(self, num_vertices, num_communities):
        self.num_vertices = num_vertices  # number of unique vertices
        self.num_communities = num_communities  # number of communities
        self.community_labels = [np.random.randint(0,num_communities) for i in range(num_vertices)]

    def get_matrix(self, cluseter_p):
        #p_matrix can be changed also 
        p_matrix = []

        cluseter_p = cluseter_p
        other_p = (1-cluseter_p)/(self.num_communities-1)

        for i in range(self.num_communities):
            row = []
            for j in range(self.num_communities):
                if(i == j):
                    row.append(cluseter_p)
                else:
                    row.append(other_p)
            
            p_matrix.append(row)

        model = SBM(self.num_vertices, self.num_communities, self.community_labels, p_matrix)
        matrix = model.block_matrix
        for i in range(0,self.num_vertices):
            matrix[i,i] = 0
            for j in range(0,self.num_vertices):
                if matrix[i,j] == 1:
                    matrix[j,i] = 1

        return matrix, self.community_labels


