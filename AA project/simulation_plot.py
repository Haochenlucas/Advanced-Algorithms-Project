import pickle
import os
import matplotlib.pyplot as plt

instance_list = list()
ratio_list = list()
rt_x_list = list()
rt_y_list = list()
score_list = list()
Q_list = list()

with open("./simulation_data/ratio_list.txt", "rb") as fp:   #Pickling
    ratio_list = pickle.load(fp)
with open("./simulation_data/rt_x_list.txt", "rb") as fp:   #Pickling
    rt_x_list = pickle.load(fp)
with open("./simulation_data/rt_y_list.txt", "rb") as fp:   #Pickling
    rt_y_list = pickle.load(fp)
with open("./simulation_data/score_list.txt", "rb") as fp:   #Pickling
    score_list = pickle.load(fp)
with open("./simulation_data/Q_list.txt", "rb") as fp:   #Pickling
    Q_list = pickle.load(fp)
with open("./simulation_data/instance_list.txt", "rb") as fp:   #Pickling
    instance_list = pickle.load(fp)


# relationship plot
plt.plot(rt_x_list,rt_y_list,'ro')
plt.axis([0,5,0,5])
plt.xlabel('top k - 1 eigenvalue / top k eigenvalue')
plt.ylabel('top k eigenvalue/ top k + 1 eigenvalue')
plt.title("relationship between top k - 1 eigenvalue / top k eigenvalue' \n and top k eigenvalue/ top k + 1 eigenvalue")
plt.show()

# ratio plot
plt.plot(instance_list,ratio_list,'ro')
plt.axis([0,30,0,5])
plt.xlabel('simulations')
plt.ylabel('ratio')
plt.title("ratio between best number of eigen_vector \nand best number of cluster")
plt.show()

print("Finished")
