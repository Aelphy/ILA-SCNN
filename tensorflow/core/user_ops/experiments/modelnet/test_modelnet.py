import tensorflow as tf
import modelnet_models as models
import numpy as np


model_location = '/tmp/modelnet10_8.meta'

model_location = '/tmp/modelnet10_8'
dim = 3 
approx = True
res = 3 
rho_data = 1. / res 
batch_size = 1 
tensor_in_sizes_=[batch_size, res, res, res, 1] #[batch, depth, height, width, in_channels]
pooling_sizes = [1,2,2,2,1]
nr_batchs = 10
batch_label_sizes = [batch_size, 10] 


tensor_in_sizes = np.array(tensor_in_sizes_, dtype=np.int64)

sparse_data = tf.sparse_placeholder(tf.float32, shape=tensor_in_sizes, name="sparse_placeholder")


var_list = []

#initialize graph

dense_labels = tf.placeholder(tf.float32, shape=batch_label_sizes, name="labels_placeholder")
sd_loss = models.model_modelnet10_8(sparse_data, tensor_in_sizes, var_list, train = True, train_labels = dense_labels, approx = approx)
sd_train_op = tf.train.AdagradOptimizer(0.1)
sd_train =  sd_train_op.minimize(sd_loss)
sd_grads = sd_train_op.compute_gradients(sd_loss)


saver = tf.train.Saver(var_list=tf.trainable_variables())
sess=tf.Session()    
#First let's load meta graph and restore weights

saver.restore(sess,model_location)

print(sess.run('sc1/filter_weights:0'))
