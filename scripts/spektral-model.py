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

from plot_test import plot_test
from plot import plot
from datasets import Vikrant_test, PCAD_test, PCAD_test_common, PCAD_test_not_common
from mymodel import Net

#################################################################
#  This is the main routine to train graph neural network.      #
#  "dataset" is our primary large dataset (12318 entries).      #
#  3 more small test sets are chosen to test the performance.   #
#  DisjointLoader is used to load the datasets. Only the        #
#  is allowed to be shuffled as the rest do not need to. Adam   #
#  is our optimizer and MAE is the loss function.               #
#                                 Vikrant Tripathy (08/01/23)   #
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

model = Net()
optimizer = Adam(learning_rate=learning_rate)
loss_fn = MeanAbsoluteError()
################################################################################
# Fit model
################################################################################
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
  with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    loss = loss_fn(target, predictions) + sum(model.losses)
  gradients = tape.gradient(loss, model.trainable_variables)

  model.summary()

#  net_parameters(model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  acc = tf.reduce_mean(mean_absolute_error(target, predictions))
  return loss, acc

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

epoch = step = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience
results = []

history = {'loss':[],'acc':[],'val_loss':[],'val_acc':[]}

for batch in loader_tr:
  step += 1
  loss, acc = train_step(*batch)
  results.append((loss, acc))
  if step == loader_tr.steps_per_epoch:
    step = 0
    epoch += 1

    # Compute validation loss and accuracy
    (val_loss, val_acc), Index_va, target_va, pred_va = evaluate(loader_va,dataset,idx_va)
    print(
        "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val loss: {:.3f} - Val acc: {:.3f}".format(
            epoch, *np.mean(results, 0), val_loss, val_acc
        )
    )

    history['loss'].append(np.mean(results,0)[0])
    history['acc'].append(np.mean(results,0)[1])
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Check if loss improved for early stopping
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      patience = es_patience
      print("New best val_loss {:.3f}".format(val_loss))
      best_weights = model.get_weights()
    else:
      patience -= 1
      if patience == 0:
        print("Early stopping (best val_loss: {})".format(best_val_loss))
        break
    results = []

fig, ax= plt.subplots()
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2,length=5)
ax.plot(history['acc'])
ax.plot(history['val_acc'])
plt.xticks(fontsize=12,fontweight='bold')
plt.yticks(fontsize=12,fontweight='bold')
ax.set_title('model accuracy', fontsize=12, fontweight='bold')
ax.set_ylabel('accuracy (eV)', fontsize=12, fontweight='bold')
ax.set_xlabel('epoch', fontsize=12, fontweight='bold')
ax.legend(['train', 'validation'], loc='upper right')
fig.savefig('training.png')

annot=False
regcol='b-'

################################################################################
# Evaluate model
################################################################################
print("Testing model")
model.set_weights(best_weights)  # Load best model
(test_loss, test_acc), Index_test, target_test, pred_test = evaluate(loader_te,dataset,idx_te)
print("Done. Test loss: {:.4f}. Test acc: {:.2f}".format(test_loss, test_acc))

with open('../data/sol.pkl','rb') as fh:
  Sol=pickle.load(fh)

fig, ax= plt.subplots()
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2,length=5)
plot_test(np.asarray(target_test),np.asarray(pred_test),'Experimental Emission (eV)','Predicted Emission (eV)','Predicted vs Experiment',Sol,regcol,fig,ax)
fig.savefig('Predicted-vs-Experiment-test.png')

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

model.save("saved_model")

