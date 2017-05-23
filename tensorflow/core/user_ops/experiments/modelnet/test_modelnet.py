import tensorflow as tf
import modelnet_models as models
import numpy as np
import time
from read_modelnet_models import ModelnetReader
from eval_confusion_matrix import ConfusionMatrixEval

data_location = '/home/thackel/Desktop/ModelNet10'
model_location = '/home/thackel/cnn_models/modelnet10_8x_14'
dim = 3 
approx = True
res = 8 
rho_data = 1. / res 
batch_size = 43
tensor_in_sizes_=[batch_size, res, res, res, 1] #[batch, depth, height, width, in_channels]
pooling_sizes = [1,2,2,2,1]
num_classes = 10

batch_label_sizes = [batch_size, num_classes] 


tensor_in_sizes = np.array(tensor_in_sizes_, dtype=np.int64)

sparse_data = tf.sparse_placeholder(tf.float32, shape=tensor_in_sizes, name="sparse_placeholder")


var_list = []

#initialize graph

dense_labels = tf.placeholder(tf.float32, shape=batch_label_sizes, name="labels_placeholder")
sd_loss = models.model_modelnet10_8(sparse_data, tensor_in_sizes, var_list, train = False, train_labels = dense_labels, approx = approx)


saver = tf.train.Saver(var_list=tf.trainable_variables())

config = tf.ConfigProto(
  device_count = {'GPU': 0}
)


sess=tf.Session(config=config)    
#First let's load meta graph and restore weights

saver.restore(sess,model_location)

writer = tf.summary.FileWriter("/tmp/test", sess.graph)
reader = ModelnetReader(data_location, res, 0, batch_size, train=False)
reader.init()
reader.start()
has_data = True
av_loss = 0 
batches = 0

while has_data:
  t1 = time.time()
  [batch, has_data] = reader.next_batch()
  reader.start()
  t2 = time.time()
  print("time: ", t2 - t1) 
  values_ = np.array(batch[1], dtype=np.float32)
  indices_ = np.array(batch[0], dtype =np.int64)
  feed_dict={sparse_data: tf.SparseTensorValue(indices_, values_, batch[2]), dense_labels: batch[3]}
  #perform training
  loss_val = sess.run(sd_loss, feed_dict=feed_dict)
  predictions = sess.run(tf.argmax(loss_val, axis=1))
  actual = sess.run(tf.argmax(batch[3], axis=1))
  batch_conf_matrix = conf_matrix = sess.run(tf.confusion_matrix(actual, predictions, num_classes=num_classes))
  if batches == 0:
    all_conf_mat = batch_conf_matrix
  else:
    all_conf_mat = np.add(batch_conf_matrix, all_conf_mat)
  batches = batches + 1  
print("confusion matrix: ", all_conf_mat)

cme = ConfusionMatrixEval(10)
cme.set_matrix(all_conf_mat)
oa = cme.get_overall_accuracy()
print("Overall Accuracy: ", oa)
iou = cme.get_intersection_union_per_class()
print("Intersection over Union per Class: ", iou)
aiou = cme.get_average_intersection_union()
print("Average Intersection over Union: ", aiou)


