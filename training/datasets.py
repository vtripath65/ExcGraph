import pickle
from spektral.data import Dataset, Graph
import numpy as np

############################################################
#  Open the files containing the information of the        #
#  datasets. Create graphs from the informations in        #
#  x(node), a(adjacency), and y (target) values. Here,     #
#  I have more information in the data file for use        #
#  in other architectures. In the current architecture     #
#  the edge properties are not used. Tests showed that     #
#  using edge features increased the outside test set      #
#  error. The idex, calc and sim are used to visulize      #
#  the predicted results. y1 is experimental emission      #
#  max and y2 is delta between experiment and mom          #
#  emission. All energies are in eV.                       #
#                           Vikrant Tripathy (08/01/2023)  #
############################################################

class Vikrant_test(Dataset):
  def read(self):
    file=open('../data/features_with_aromaticity_simplified.pkl','rb')
    data=pickle.load(file)
# y1 is experimental and y2 is (exp-mom). We are limiting ourselves to emission.
    return [Graph(x=np.array(list(x.values())), a=adj, y=y1, target=y1, index=index) for index, x, adj, y1 in data]

class PCAD_test(Dataset):
  def read(self):
    file=open('../data/PCAD_features_with_aromaticity_simplified.pkl','rb')
    data=pickle.load(file)
    return [Graph(x=np.array(list(x.values())), a=adj, y=y1, target=y1, index=index, sim=sim) for index, x, adj, y1, sim in data]

class PCAD_test_common(Dataset):
  def read(self):
    file=open('../data/PCAD_features_with_aromaticity_simplified_common.pkl','rb')
    data=pickle.load(file)
    return [Graph(x=np.array(list(x.values())), a=adj, y=y1, target=y1, index=index, sim=sim) for index, x, adj, y1, sim in data]

class PCAD_test_not_common(Dataset):
  def read(self):
    file=open('../data/PCAD_features_with_aromaticity_simplified_not_common.pkl','rb')
    data=pickle.load(file)
    return [Graph(x=np.array(list(x.values())), a=adj, y=y1, target=y1, index=index, sim=sim) for index, x, adj, y1, sim in data]

