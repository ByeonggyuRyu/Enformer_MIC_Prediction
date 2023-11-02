import os
import re
import sys
import math
import pickle
import random
import enformer
import numpy as np
import sonnet as snt
from time import time
from tqdm import tqdm
import tensorflow as tf
from datetime import timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3, 4, 5, 6, 7'

gpus = tf.config.list_physical_devices("GPU")
assert gpus
for i in range(len(gpus)):
    tf.config.experimental.set_memory_growth(gpus[i], True)

strategy = snt.distribute.Replicator(devices=["/GPU:2", "/GPU:3", "/GPU:4", "/GPU:5", "/GPU:6", "/GPU:7"])

BATCH_SIZE = 16

warmup_init_lr = 0.00000000000001
train_init_lr = 0.0002

train_log_path = sys.argv[1] # './results/train_log.txt' 
test_result_path = sys.argv[2] #'./results/test_result.txt'  
model_path = sys.argv[3] #'./results/model' 
train_dataset_path = sys.argv[4] # './data/tfrecord_data/train_data.tfrecord' 
val_dataset_path = sys.argv[5] # './data/tfrecord_data/val_data.tfrecord'
test_dataset_path = sys.argv[6] # './data/tfrecord_data/test_data.tfrecord'

########################################################################################

def build_dataset_from_tfrecord(filename: str) -> tf.data.Dataset:
    """
    Build tf.data.Dataset from tfrecord file and .info file.
    Args:
        filename (str): filename of tfrecord
    Returns:
        tf.data.Dataset: Dataset of the data contained in the tfrecord
    """

    info_filename = f"{filename}.info"
    try:
      assert os.path.exists(info_filename)
    except:
        FileNotFoundError
    [dtypes, shapes] = pickle.load(open(info_filename, "rb"))

    def parse_tfrecord(example: tf.train.Example):
      feature_desc = dict()
      for key in dtypes.keys():
        if dtypes[key] == np.float32:
          feature_desc[key] = tf.io.VarLenFeature(tf.float32)
        elif dtypes[key] == np.int64:
          feature_desc[key] = tf.io.VarLenFeature(tf.int64)
        else:
          raise NotImplementedError(
            f"dtype {dtypes[key]} is not currently supported"
          )
      features = tf.io.parse_single_example(example, feature_desc)

      sample = dict()
      for key in shapes.keys():
        sample[key] = tf.reshape(tf.sparse.to_dense(features[key]), shapes[key])
      return sample

    raw_dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=32)
    return raw_dataset.map(parse_tfrecord)

####################################################################################

def is_predicted(inputs, labels, mode):
  if mode == 'val':
    outputs = model.predict_on_batch(inputs)["MIC"]
  elif mode == 'test':
    outputs = loaded_model.predict_on_batch(inputs)["MIC"]
  else:
    raise ValueError(f'Invalid mode: {mode}.')
  raw = tf.cast(tf.equal(tf.argmax(labels, axis=1), tf.argmax(outputs, axis=1)), dtype=tf.int32)
  raw_wrong = tf.cast(tf.not_equal(tf.argmax(labels, axis=1), tf.argmax(outputs, axis=1)), dtype=tf.int32)
  onetier = tf.cast(tf.cast(tf.abs(tf.argmax(labels, axis=1) -  tf.argmax(outputs, axis=1)), dtype=tf.int32) <= 1, dtype=tf.int32)
  return raw, raw_wrong, onetier

@tf.function
def tf_evaluate(inputs, labels, mode):
  per_replica_raw, per_replica_raw_wrong, per_replica_onetier = strategy.run(is_predicted, args=(inputs, labels, mode))
  return strategy.reduce("sum", per_replica_raw, axis=0), strategy.reduce("sum", per_replica_raw_wrong, axis=0), strategy.reduce("sum", per_replica_onetier, axis=0)

def evaluate(dataset, mode, size):
  raw_correct = 0
  raw_incorrect = 0
  onetier_correct = 0

  for batch in dataset:
    raw, raw_wrong, onetier = tf_evaluate(batch['seq_mat'], batch['mic_label'], mode)
    raw_correct += raw.numpy()
    raw_incorrect += raw_wrong.numpy()
    onetier_correct += onetier.numpy()

  assert raw_correct + raw_incorrect == size
  return raw_correct / size, onetier_correct / size

####################################################################################

def step(inputs, labels):
  """Performs a single training step, returning the cross-entropy loss."""
  with tf.GradientTape() as tape:
    outputs = model(inputs, is_training=True)["MIC"]
    ########## For strict labeling ####################
    labels = tf.one_hot(tf.argmax(labels, 1), depth=11)
    ###################################################
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, outputs))

  grads = tape.gradient(loss, model.trainable_variables)
  replica_ctx = tf.distribute.get_replica_context()
  grads = replica_ctx.all_reduce("mean", grads)

  optimizer.apply(grads, model.trainable_variables)
  return loss

@tf.function
def train_step(inputs, labels):
  per_replica_loss = strategy.run(step, args=(inputs, labels))
  return strategy.reduce("sum", per_replica_loss, axis=None)

def train_epoch(dataset, epoch):
  """Performs one epoch of training, returning the mean cross-entropy loss."""
  total_loss = 0.0
  num_batches = 0

  if epoch < 0 :
    for batch in dataset:
      total_loss += train_step(batch['seq_mat'], batch['mic_label']).numpy()
      num_batches += 1
      learning_rate.assign(warmup_init_lr + (((train_init_lr - warmup_init_lr) / 3304) * ((epoch + 2) * 1652 + num_batches)))
      print('epoch: ', epoch, 'step: ', num_batches, 'loss: ', total_loss/num_batches, learning_rate)
  else :
    for batch in dataset:
      total_loss += train_step(batch['seq_mat'], batch['mic_label']).numpy()
      num_batches += 1
      print('epoch: ', epoch, 'step: ', num_batches, 'loss: ', total_loss/num_batches)

  return total_loss / num_batches

####################################################################################

def load_train_data(train_size):
  train_dataset = build_dataset_from_tfrecord(train_dataset_path).take(int((train_size // 8) * 8)).batch(BATCH_SIZE)
  train_dataset = strategy.experimental_distribute_dataset(train_dataset)
  return train_dataset

def load_val_data():
  val_dataset = build_dataset_from_tfrecord(val_dataset_path)
  val_size = 0
  for data_point in val_dataset:
    val_size += 1
  val_dataset = val_dataset.batch(BATCH_SIZE)
  val_dataset = strategy.experimental_distribute_dataset(val_dataset)
  return val_dataset, val_size

def load_test_data():
  test_dataset = build_dataset_from_tfrecord(test_dataset_path)
  test_size = 0
  for data_point in test_dataset:
    test_size += 1
  test_dataset = test_dataset.batch(BATCH_SIZE)
  test_dataset = strategy.experimental_distribute_dataset(test_dataset)
  return test_dataset, test_size

####################################################################################

if __name__=='__main__':

    with strategy.scope():
      learning_rate = tf.Variable(train_init_lr, trainable=False, name='learning_rate')
      optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

      model = enformer.Enformer(channels=192,
                              num_heads=8,
                              num_transformer_layers=2,
                              pooling_type='max')

    val_dataset, val_size = load_val_data()
    train_dataset = load_train_data(29371 - val_size)

    learning_rate.assign(warmup_init_lr)
    for epoch in [-2, -1]:
      train_epoch(train_dataset, epoch)
    
    learning_rate.assign(train_init_lr)
    reducelr_lst = []
    early_stopping_lst = []
    start_time = time()
    for epoch in range(100):
      print("Training epoch", epoch, "...", end="\n")
      train_loss_per_epoch = train_epoch(train_dataset, epoch)
      raw_acc_per_epoch, one_tier_acc_per_epoch = evaluate(val_dataset, 'val', val_size)
      with open(train_log_path, 'a') as f:
        f.write('epoch_' + str(epoch) + ', ' + str(timedelta(seconds=(time() - start_time))) + ', ' + 'train_loss:' +str(train_loss_per_epoch) + ', ' 
                + 'raw_val_acc:' + str(raw_acc_per_epoch)+ ', ' + '1-tier_val_acc:'+str(one_tier_acc_per_epoch)  + '\n')
      print("train_loss :=", train_loss_per_epoch)
      print("raw_acc :=", raw_acc_per_epoch)
      print("1-tier_acc :=", one_tier_acc_per_epoch)
      reducelr_lst.append(one_tier_acc_per_epoch)
      early_stopping_lst.append(one_tier_acc_per_epoch)
      if(len(early_stopping_lst) >= 20):
        if(early_stopping_lst[-1] == max(early_stopping_lst)):
          tf.saved_model.save(model, model_path)
      if(len(early_stopping_lst) >= 16):
        if(max(early_stopping_lst) not in early_stopping_lst[-16:]):
          break
      if(len(reducelr_lst) >= 4):
        if (max(reducelr_lst) not in reducelr_lst[-4:]):
          learning_rate.assign(learning_rate / 2)
          with open(train_log_path, 'a') as f:
            f.write('learning rate reduced by 1/2 \n')
          reducelr_lst = []

    test_dataset, test_size = load_test_data()
    loaded_model = tf.saved_model.load(model_path)

    test_raw_acc, test_one_tier_acc = evaluate(test_dataset, 'test', test_size)
    with open(test_result_path, 'a') as f:
      f.write('raw_val_acc:' + str(test_raw_acc)+ ', ' + '1-tier_val_acc:'+str(test_one_tier_acc)  + '\n')