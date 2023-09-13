import pickle
import numpy as np
import matplotlib.pyplot as plt

from spektral.data import DisjointLoader
from spektral.transforms import GCNFilter

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_absolute_error
from tensorflow.keras.losses import MeanAbsoluteError

from plot import plot
from datasets import Vikrant_test, PCAD_test, PCAD_test_common, PCAD_test_not_common

#################################################################
#  This is the main routine to load the saved model 08_01_23.   #
#                                 Vikrant Tripathy (09/13/23)   #
#################################################################

dataset = Vikrant_test(transforms=GCNFilter())
PCAD_dataset = PCAD_test(transforms=GCNFilter())
PCAD_dataset_common = PCAD_test_common(transforms=GCNFilter())
PCAD_dataset_not_common = PCAD_test_not_common(transforms=GCNFilter())

F = dataset.n_node_features
S = dataset.n_edge_features
n_out = dataset.n_labels

idxs = np.random.permutation(len(dataset))
split_va, split_te = int(0.8 * len(dataset)), int(0.9 * len(dataset))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
dataset_tr, dataset_va, dataset_te = dataset[idx_tr], dataset[idx_va], dataset[idx_te]

idx_PCAD = np.array(range(len(PCAD_dataset)))
idx_PCAD_common = np.array(range(len(PCAD_dataset_common)))
idx_PCAD_not_common = np.array(range(len(PCAD_dataset_not_common)))

learning_rate = 1e-3
epochs = 400
seed = 0
es_patience = 50
batch_size = 32

tf.random.set_seed(seed=seed)

loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(dataset_va, batch_size=batch_size, shuffle=False)
loader_te = DisjointLoader(dataset_te, batch_size=batch_size, shuffle=False)

loader_PCAD = DisjointLoader(PCAD_dataset, batch_size=batch_size, shuffle=False)
loader_PCAD_common = DisjointLoader(PCAD_dataset_common, batch_size=batch_size, shuffle=False)
loader_PCAD_not_common = DisjointLoader(PCAD_dataset_not_common, batch_size=batch_size, shuffle=False)

loss_fn = MeanAbsoluteError()

def evaluate(loader,dataset,idx):
  output = []
  step = Ind = 0
  Index_list = []
  pred_list = []
  target_list = []
  while step < loader.steps_per_epoch:
    step += 1
    inputs, target = loader.__next__()

    Ind1=Ind+inputs[2][-1]+1
    for x in dataset[idx[Ind:Ind1]]:
      Index_list.append(x.index)
      target_list.append(x.target)

    pred = model(inputs, training=False)
    pred_list = pred_list+list(pred.numpy().flatten())

    outs = (
        loss_fn(target, pred),
        tf.reduce_mean(mean_absolute_error(target, pred)),
        len(target),  # Keep track of batch size
    )
    output.append(outs)
    if step == loader.steps_per_epoch:
      pred_list = list(np.array(pred_list))
      output = np.array(output)
      return np.average(output[:, :-1], 0, weights=output[:, -1]), Index_list, target_list, pred_list
    Ind=Ind1

def evaluate_pcad(loader,dataset,idx):
  output = []
  step = Ind = 0
  Index_list = []
  pred_list = []
  target_list = []
  sim_list=[]
  while step < loader.steps_per_epoch:
    step += 1
    inputs, target = loader.__next__()

    Ind1=Ind+inputs[2][-1]+1
    for x in dataset[idx[Ind:Ind1]]:
      Index_list.append(x.index)
      target_list.append(x.target)
      sim_list.append(x.sim)

    pred = model(inputs, training=False)
    pred_list = pred_list+list(pred.numpy().flatten())

    outs = (
        loss_fn(target, pred),
        tf.reduce_mean(mean_absolute_error(target, pred)),
        len(target),  # Keep track of batch size
    )
    output.append(outs)
    if step == loader.steps_per_epoch:
      pred_list = list(np.array(pred_list))
      output = np.array(output)
      return np.average(output[:, :-1], 0, weights=output[:, -1]), Index_list, target_list, pred_list, sim_list
    Ind=Ind1

annot=False
regcol='b-'

################################################################################
# Evaluate model
################################################################################
model=load_model("saved_model")

print("Testing PhotoChecmCAD model")
(pcad_loss, pcad_acc), index_pcad, target_pcad, pred_pcad = evaluate(loader_PCAD,PCAD_dataset,idx_PCAD)
print("done. pcad loss: {:.4f}. pcad acc: {:.2f}".format(pcad_loss, pcad_acc))

fig, ax= plt.subplots()
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2,length=5)
plot(np.asarray(target_pcad),np.asarray(pred_pcad),'Experimental Emission (eV)','Predicted Emission (eV)','Predicted vs Experiment',np.asarray(index_pcad),regcol,annot,fig,ax)
fig.savefig('Predicted-vs-Experiment.png')

annot=True

fig, ax= plt.subplots()
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2,length=5)
plot(np.asarray(target_pcad),np.asarray(pred_pcad),'Experimental Emission (eV)','Predicted Emission (eV)','Predicted vs Experiment',np.asarray(index_pcad),regcol,annot,fig,ax)
fig.savefig('Predicted-vs-Experiment-annotate.png')

print("Testing PhotoChecmCAD model for common")
(pcad_loss_common, pcad_acc_common), index_pcad_common, target_pcad_common, pred_pcad_common = evaluate(loader_PCAD_common,PCAD_dataset_common,idx_PCAD_common)
print("done. pcad loss: {:.4f}. pcad acc: {:.2f}".format(pcad_loss_common, pcad_acc_common))

annot=False

fig, ax= plt.subplots()
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2,length=5)
plot(np.asarray(target_pcad_common),np.asarray(pred_pcad_common),"Experimental Emission (eV)","Predicted Emission (eV)","Predicted vs Experiment",np.asarray(index_pcad_common),regcol,annot,fig,ax)
fig.savefig("Predicted-vs-Experiment_common.png")

print("Testing PhotoChecmCAD model for not common")
(pcad_loss_not_common, pcad_acc_not_common), index_pcad_not_common, target_pcad_not_common, pred_pcad_not_common, sim_pcad_not_common = evaluate_pcad(loader_PCAD_not_common,PCAD_dataset_not_common,idx_PCAD_not_common)
print("done. pcad loss: {:.4f}. pcad acc: {:.2f}".format(pcad_loss_not_common, pcad_acc_not_common))

annot=False

fig, ax= plt.subplots()
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2,length=5)
plot(np.asarray(target_pcad_not_common),np.asarray(pred_pcad_not_common),"Experimental Emission (eV)","Predicted Emission (eV)","Predicted vs Experiment",np.asarray(index_pcad_not_common),regcol,annot,fig,ax)
fig.savefig("Predicted-vs-Experiment_not_common.png")

fig, ax= plt.subplots()
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2,length=5)
plot(np.asarray(sim_pcad_not_common),np.abs(np.subtract(np.asarray(pred_pcad_not_common),np.asarray(target_pcad_not_common))),"Max similarity","Absolute Error in Emission (eV)","Absolute Error vs Similarity",np.asarray(index_pcad_not_common),regcol,annot,fig,ax)
fig.savefig("Error-vs-Similarity_not_common.png")

annot=True

fig, ax= plt.subplots()
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2,length=5)
plot(np.asarray(target_pcad_not_common),np.asarray(pred_pcad_not_common),"Experimental Emission (eV)","Predicted Emission (eV)","Predicted vs Experiment",np.asarray(index_pcad_not_common),regcol,annot,fig,ax)
fig.savefig("Predicted-vs-Experiment-annotate_not_common.png")

fig, ax= plt.subplots()
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2,length=5)
plot(np.asarray(sim_pcad_not_common),np.abs(np.subtract(np.asarray(pred_pcad_not_common),np.asarray(target_pcad_not_common))),"Max similarity","Absolute Error in Emission (eV)","Absolute Error vs Similarity",np.asarray(index_pcad_not_common),regcol,annot,fig,ax)
fig.savefig("Error-vs-Similarity-annotate_not_common.png")

