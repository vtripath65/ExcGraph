from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization
from spektral.layers import GCNConv, GlobalAttentionPool

#######################################################
#  The graph network architecture is defined here.    #
#  BatchNormalization is used before and after        #
#  every convolution layer to minimize internal       #
#  covariate shift. The Graph Convolutional Network   #
#  is used to perform convoltion on the graphs. The   #
#  GlobalAttentionPool layer extracts a global        #
#  molecular fingerprint. This is then fed through    #
#  a two-layer feed forward neural network to get     #
#  the predicted value.                               #
#                        Vikrant Tripathy (08/01/23)  #
#######################################################

class Net(Model):
  def __init__(self):
    super().__init__()
    self.batchnorm1 = BatchNormalization()
    self.conv1 = GCNConv(64, activation="relu")
    self.batchnorm2 = BatchNormalization()
    self.conv2 = GCNConv(64, activation="relu")
    self.batchnorm3 = BatchNormalization()
    self.global_pool = GlobalAttentionPool(32)
    self.dense1 = Dense(64, activation="relu")
    self.dense2 = Dense(1)

  def call(self, inputs):
    x, a, i = inputs
    x = self.batchnorm1(x)
    x = self.conv1([x, a])
    x = self.batchnorm2(x)
    x = self.conv2([x, a])
    x = self.batchnorm3(x)
    x = self.global_pool([x,i])
    x = self.dense1(x)
    output = self.dense2(x)

    return output

