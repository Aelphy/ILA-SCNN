import tensorflow as tf
import modelnet_models as models
import numpy as np
import time
from read_modelnet_models import ModelnetReader
from eval_confusion_matrix import ConfusionMatrixEval
from sparse_module import sparse_nn_ops as sc_module

data_location = '/home/thackel/Desktop/ModelNet10'
model_location = '/home/thackel/cnn_models/modelnet10_8_sd1_17'
dim = 3 
approx = False
res = 8 
rho_data = 1. / res 
batch_size = 43
tensor_in_sizes_=[batch_size, res, res, res, 1] #[batch, depth, height, width, in_channels]
pooling_sizes = [1,2,2,2,1]
reader = ModelnetReader(data_location, res, 0, batch_size, train=False)
num_classes = reader.getNumClasses()

batch_label_sizes = [batch_size, num_classes] 


tensor_in_sizes = np.array(tensor_in_sizes_, dtype=np.int64)

sparse_data = tf.sparse_placeholder(tf.float32, shape=tensor_in_sizes, name="sparse_placeholder")


var_list = []

#initialize graph

dense_labels = tf.placeholder(tf.float32, shape=batch_label_sizes, name="labels_placeholder")
sd_loss = models.model_modelnet10_8(sparse_data, tensor_in_sizes, var_list, train = False, train_labels = dense_labels, approx = approx, num_classes = num_classes)


saver = tf.train.Saver(var_list=tf.trainable_variables())

config = tf.ConfigProto(
  device_count = {'GPU': 0}
)


sess=tf.Session(config=config)    
#First let's load meta graph and restore weights

saver.restore(sess,model_location)

writer = tf.summary.FileWriter("/tmp/test", sess.graph)
reader.init()
reader.start()
has_data = True
av_loss = 0 
batches = 0

all_conf_mat = tf.zeros([num_classes, num_classes], dtype=tf.int32)

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
  all_conf_mat = all_conf_mat  + tf.confusion_matrix(actual, predictions, num_classes=num_classes)
  batches = batches + 1

[res_conf_mat, [iou, aiou, oa]] = sess.run([all_conf_mat, sc_module.evaluate_confusion_matrix(all_conf_mat)])

print("confusion matrix: ", res_conf_mat)

print("iou:", iou)
print("aiou:", aiou)
print("oa:", oa)
