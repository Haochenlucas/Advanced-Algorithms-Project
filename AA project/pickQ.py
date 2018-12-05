from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx
import pandas
import os 
from mpl_toolkits.mplot3d import Axes3D
import math
from simulationMatrix import SBM_Matrix
import pickle

# print(os.getcwd())
os.chdir("../")

def draw_clusters(matrix, labels):
    number_of_cluster = len(np.unique(labels))
    cmap = get_cmap(number_of_cluster)
    G = nx.from_numpy_matrix(matrix) 

    #plt.figure(figsize=(200,200))
    for c in range(0,number_of_cluster):
        node_list = np.where(labels == c)[0].tolist()
        nx.draw_networkx_nodes(G,pos=nx.spectral_layout(G),nodelist= node_list,
                        node_color= cmap(c),
                        node_size=50,
                        alpha=0.8)
    #G.edges(node)

    # edges
    # nx.draw_networkx_edges(G,pos = nx.spectral_layout(G),
    #                     edgelist=G.edges(),
    #                     width=1,alpha=0.5)

    plt.axis('off')

def draw_3d(st):
    st_use = st[2:40,2:50]
    x = list()
    y = list()
    z = list()
    for i in range(st_use.shape[0]):
        for j in range(st_use.shape[1]):
            x.append(i + 5)
            y.append(j + 10)
            z.append(st_use[i,j])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c="r")
    ax.set_xlabel('number of eigen_vector')
    ax.set_ylabel('number of cluster')
    ax.set_zlabel('Modularity')

def Modularity(label,weight_mat):
    s = np.unique(label)
    l = len(s)
    if len(np.unique(weight_mat.diagonal())) != 1:
        print("something goes wrong in weight mat")
        return 0
    total = np.sum(weight_mat)  
    result = 0 
    for i in range(0,l):
        #print(i)
        #pdb.set_trace()
        row = np.array(np.where(label == s[i]))
        length = len(np.transpose(row))
        r1 = np.reshape(row,length)
        small_mat = weight_mat[np.ix_(r1,r1)]
        In_Cluster = np.sum(small_mat)
        #print(In_Cluster/total)
        #row_out = np.array(np.where(label != s[i])) 
        #length_2 = len(np.transpose(row_out))
        r2 = range(0,weight_mat.shape[0])
        out_mat = weight_mat[np.ix_(r1,r2)]
        Out_Cluster = np.sum(out_mat) 
        #print(Out_Cluster/total)
        result = result + In_Cluster/total - (Out_Cluster/total)**2
    return result    

def pickQ(weight_mat,k):
    m_mat = np.diag(np.sum(weight_mat,axis=1))
    M_mat = np.matmul(np.linalg.inv(m_mat),weight_mat)
    w,v = linalg.eigh(M_mat)
    max_cluster = min(50,weight_mat.shape[0])
    result_mat = np.zeros(shape = (k,max_cluster))
    a = np.abs(w)
    a.sort()
    c = np.abs(w)
    #pdb.set_trace()
    eigen_mat = np.zeros(shape = (weight_mat.shape[0],k)) ### at most I use k eigen-vectors 
    for s in range(0,k):
        #pdb.set_trace()
        index_picked = np.where(c == a[weight_mat.shape[0] - 1 - s])
        q = int(index_picked[0])
        eigen_mat[:,s] = v[:,q]
    # i is the number of the cluster 
    #pdb.set_trace()
    for i in range(2,max_cluster):
        ## j is number of eigen_vector picked 
        for j in range(1,k):
            eigen_mat_j = eigen_mat[:,0:j]
            kmeans = KMeans(n_clusters=i, random_state=0).fit(eigen_mat_j)
            result_mat[j,i] = Modularity(kmeans.labels_, weight_mat)
    max_index = np.unravel_index(np.argmax(result_mat),(k,50))
    nc = max_index[1]
    nv = max_index[0]
    eigen_mat_final = eigen_mat[:,0:nv]
    kmeans_final = KMeans(n_clusters=nc,random_state = 0).fit(eigen_mat_final)
    return result_mat, kmeans_final.labels_        

# for color choosing
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

# wordNet graph
wordNet_data = np.loadtxt(os.getcwd() + "/wordNet.out")
st_wordNet,labels_wordNet = pickQ(wordNet_data,40)
draw_clusters(wordNet_data, labels_wordNet)
plt.show()
plt.close()
draw_3d(st_wordNet)
plt.show()
plt.close()


# NCAA graph
NCAA_data = np.loadtxt("./NCAA.out")
st_NCAA,labels_NCAA = pickQ(NCAA_data,40)
draw_clusters(NCAA_data, labels_NCAA)
plt.show()
plt.close()
draw_3d(st_NCAA)
plt.show()
plt.close()

# Simulation pick Q data generate
vertices = [100,250,500]
clusters = [5,7,10]
p_edge = [0.4, 0.7, 0.9]

instance_list = list()
ratio_list = list()
rt_x_list = list()
rt_y_list = list()
score_list = list()
Q_list = list()

instance_count = 0
for vertex in vertices:
    for cluster in clusters:
        for p in p_edge:
            instance_count += 1
            instance_list.append(instance_count)

            ratio_ave_list = list()
            rt_x_ave_list = list()
            rt_y_ave_list = list()
            score_ave_list = list()
            Q_ave_list = list()

            for i in range(10):
                sim_matrix = SBM_Matrix(vertex, cluster)
                matrix, community_label = sim_matrix.get_matrix(p)
                st_sim,labels_sim = pickQ(matrix,40)

                ## n_v best number of eigen_vector
                ## n_c best number of cluster
                n_v, n_c = np.unravel_index(st_sim.argmax(),st_sim.shape)
                ratio = n_v/n_c ## stored ratio
                ratio_ave_list.append(ratio)
                m_mat = np.diag(np.sum(matrix,axis=1))
                M_mat = np.matmul(np.linalg.inv(m_mat),matrix)
                w,v = linalg.eigh(M_mat)
                a = np.abs(w)
                ## sort eigen_values
                a.sort()
                c = np.abs(w)
                shape = matrix.shape[0] ## eigen_decompostion in fact will produce number of samples eigenvector
                #### shape == number of vertex
                #a[shape-n_v : shape] ### top k + 1 eigenvalue .... (you have picked k in your clustering)
                rt_x = 1/(a[- n_v] / a[1- n_v])  ## k-1 / k
                rt_x_ave_list.append(rt_x)
                rt_y = 1/(a[ - n_v - 1] /a[-n_v]) ## k / k+1
                rt_y_ave_list.append(rt_y)

                score = adjusted_rand_score(labels_sim,community_label)
                score_ave_list.append(score)
                Q = st_sim[n_v,n_c]
                Q_ave_list.append(Q)

            ratio_list.append(sum(ratio_ave_list)/float(10))
            rt_x_list.append(sum(rt_x_ave_list)/float(10))
            rt_y_list.append(sum(rt_y_ave_list)/float(10))
            score_list.append(sum(score_ave_list)/float(10))
            Q_list.append(sum(Q_ave_list)/float(10))

            # store data using pickle
            with open("ratio_list.txt", "wb") as fp:   #Pickling
                pickle.dump(ratio_list, fp)
            with open("rt_x_list.txt", "wb") as fp:   #Pickling
                pickle.dump(rt_x_list, fp)
            with open("rt_y_list.txt", "wb") as fp:   #Pickling
                pickle.dump(rt_y_list, fp)
            with open("score_list.txt", "wb") as fp:   #Pickling
                pickle.dump(score_list, fp)
            with open("Q_list.txt", "wb") as fp:   #Pickling
                pickle.dump(Q_list, fp)
            with open("instance_list.txt", "wb") as fp:   #Pickling
                pickle.dump(instance_list, fp)

            print(instance_count)

print("End")