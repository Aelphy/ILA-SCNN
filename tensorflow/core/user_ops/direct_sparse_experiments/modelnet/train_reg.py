import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import sparse_tools as st

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='modelnet_16_3', help='Model name: modelnet_cls or modelnet_cls_basic [default: modelnet_16_3]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--resolution', type=int, default=16, help='resolution for voxel grid [default: 16]')
parser.add_argument('--init_var', type=float, default=0.1, help='variance for initialization [default: 0.1]')
parser.add_argument('--reg_scale', type=float, default=0.1, help='regularization scale [default: 0.1]')
parser.add_argument('--reg_bias', type=float, default=0.1, help='regularization bias [default: 0.1]')
parser.add_argument('--pruning_threshold', type=float, default=0.0001, help='pruning threshold [default: 0.0001]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
RESOLUTION = FLAGS.resolution
INIT_VAR = FLAGS.init_var
REG_SCALE = FLAGS.reg_scale
REG_BIAS = FLAGS.reg_bias
PRUNING_THR = FLAGS.pruning_threshold
TENSOR_IN_SIZES = np.array([BATCH_SIZE, RESOLUTION, RESOLUTION, RESOLUTION, 1], dtype=np.int64)

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.01) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_CLASSES, TENSOR_IN_SIZES)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            #placeholders for pruning
            #print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            initializer = tf.truncated_normal_initializer(0, 0.1)
            [loss, pred, reg_ops] = MODEL.get_model(pointclouds_pl, labels_pl, is_training_pl, TENSOR_IN_SIZES, NUM_CLASSES, initializer = initializer, rscale = REG_SCALE, max_bias = REG_BIAS)
            loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.argmax(tf.to_int64(labels_pl), 1))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif OPTIMIZER == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate)
            elif OPTIMIZER == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate)
            elif OPTIMIZER == 'ftrl':
                optimizer = tf.train.FtrlOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)


        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        kernels = {}
        to_remove = {}

        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
          if 'filter' in var.name and 'Adagrad' not in var.name:
            parts = var.name.split('/')

            if parts[0] not in kernels:
              kernels[parts[0]] = {}

            if 'filter_indices' in parts[1]:
              kernels[parts[0]]['filter_indices'] = var
            if 'filter_shape' in parts[1]:
              kernels[parts[0]]['filter_shape'] = var
            if 'filter_channel_mapping' in parts[1]:
              kernels[parts[0]]['filter_channel_mapping'] = var
            if 'filter_values' in parts[1]:
              kernels[parts[0]]['filter_values'] = var

        update_ops = {}
        for layer, values in kernels.items():
          p1 = tf.placeholder(dtype=tf.float32, name='values')
          p2 = tf.placeholder(dtype=tf.int64, name='indices')
          p3 = tf.placeholder(dtype=tf.int32, name='channel_mapping')
          update_ops[layer] = {
            p1 : tf.assign(values['filter_values'], p1),
            p2 : tf.assign(values['filter_indices'], p2),
            p3 : tf.assign(values['filter_channel_mapping'], p3)
          }

        f = open(LOG_DIR + '/reg_experiment.log', 'wb')
        f.write('t_eval, t_train, delta filter weights, #filter weights, test loss, test accuracy, training loss, training accuracy\n')
        f.flush()


        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            time_count, av_loss, av_acc, sum_removed, new_weights = train_one_epoch(sess, ops, reg_ops, train_writer, kernels, update_ops, to_remove, f, run_options, run_metadata, epoch)
            ev_time, ev_loss, ev_acc = eval_one_epoch(sess, ops, test_writer)
            f.write('{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(ev_time, time_count, sum_removed, new_weights, ev_loss, ev_acc, av_loss, av_acc)+'\n')
            f.flush()
            # Save the variables to disk.
            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
            log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, reg_ops, train_writer, kernels, update_ops, to_remove, f, run_options, run_metadata, epoch):
    #g1 = str(tf.get_default_graph().as_graph_def())
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    time_count = 0
    av_loss = 0
    av_acc = 0
    sum_removed = 0
    new_weights = 0

    with tf.control_dependencies([ops['train_op']]):
      dummy = tf.constant(0)

    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,0:NUM_POINT,:]
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
        current_label = np.squeeze(current_label)
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            argmax_labels = np.zeros([BATCH_SIZE, NUM_CLASSES])
            #argmax_labels[:, this_label] = 1
            this_label = current_label[start_idx:end_idx]
            for id in range(len(this_label)):
              argmax_labels[id, this_label[id]] = 1
              #print(argmax_labels[id])
            # Augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)

            [sparse_ind, sparse_values] = provider.toVoxelGrid(jittered_data, 0, RESOLUTION)
            data = tf.SparseTensorValue(sparse_ind, sparse_values, TENSOR_IN_SIZES)
            #convert to dense if needed
            #data = st.sparse_to_dense(sparse_ind, sparse_values, TENSOR_IN_SIZES)
            time_start = time.time()
            feed_dict = {ops['pointclouds_pl']: data,
                         ops['labels_pl']: argmax_labels,
                         ops['is_training_pl']: is_training,}
            handle = sess.partial_run_setup([ops['merged'], dummy, ops['step'], ops['loss'], ops['pred']] + reg_ops, [ops['pointclouds_pl'], ops['labels_pl'], ops['is_training_pl']])
            summary, step, _, loss_val, pred_val = sess.partial_run(handle, [ops['merged'], ops['step'],
                dummy, ops['loss'], ops['pred']], feed_dict=feed_dict)
            sess.partial_run(handle, reg_ops)
            time_count += time.time() - time_start
            train_writer.add_summary(summary, step)

            pred_val = np.argmax(pred_val, 1)
            correct = (pred_val == current_label[start_idx:end_idx]).sum()
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val
        av_loss += loss_sum / float(num_batches)
        av_acc += total_correct / float(max(total_seen,1))
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(max(total_seen,1))))

    # Weights removal
    removed_weights = 0
    weights_total = 0
    for layer, values in kernels.items():
        num_filter_values = sess.run(values['filter_values'])
        current_small = np.abs(num_filter_values) < PRUNING_THR
        weights_total += (num_filter_values != -1).sum()

        if layer in to_remove:
            prev_small = to_remove[layer]
            num_filter_indices = sess.run(values['filter_indices'])
            num_filter_channel_mapping = sess.run(values['filter_channel_mapping'])
            indices_to_remove = prev_small & current_small
            removed_weights += indices_to_remove.sum()

            new_filter_values = num_filter_values[~indices_to_remove]
            new_filter_indices = num_filter_indices[~indices_to_remove]
            fill_values = -1 * np.ones(num_filter_values.shape[0] - new_filter_values.shape[0])
            fill_indices = -1 * np.ones(num_filter_indices.shape[0] - new_filter_indices.shape[0])
            to_subtract = np.zeros_like(num_filter_channel_mapping)
            for i in range(1, len(num_filter_channel_mapping)):
                to_subtract[i] = indices_to_remove[:num_filter_channel_mapping[i]].sum()

            feed_dict = {}
            for k, v in update_ops[layer].items():
              if 'values' in k.name:
                feed_dict[k] = np.hstack([new_filter_values, fill_values])
              if 'indices' in k.name:
                feed_dict[k] = np.hstack([new_filter_indices, fill_indices])
              if 'channel_mapping' in k.name:
                feed_dict[k] = num_filter_channel_mapping - to_subtract
            sess.run(update_ops[layer].values(), feed_dict=feed_dict)

        to_remove[layer] = current_small
    '''g2 = str(tf.get_default_graph().as_graph_def())
    import difflib
    expected=g1.splitlines(1)
    actual=g2.splitlines(1)
    diff=difflib.unified_diff(expected, actual)
    print(''.join(diff))'''

    print('Pruning removed {} out of {} weights'.format(removed_weights, weights_total))
    sum_removed += removed_weights
    new_weights = weights_total
    av_loss = av_loss / len(TRAIN_FILES)
    av_acc = av_acc / len(TRAIN_FILES)
    return time_count, av_loss, av_acc, sum_removed, new_weights



def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    time_count = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            argmax_labels = np.zeros([BATCH_SIZE, NUM_CLASSES])
            #argmax_labels[:, current_label[start_idx:end_idx]] = 1
            this_label = current_label[start_idx:end_idx]
            for id in range(len(this_label)):
              argmax_labels[id, this_label[id]] = 1

            [sparse_ind, sparse_values] = provider.toVoxelGrid(current_data[start_idx:end_idx, :, :], 0, RESOLUTION)
            data = tf.SparseTensorValue(sparse_ind, sparse_values, TENSOR_IN_SIZES)
            #convert to dense if needed
            #data = st.sparse_to_dense(sparse_ind, sparse_values, TENSOR_IN_SIZES)

            time_start = time.time()
            feed_dict = {ops['pointclouds_pl']: data,
                         ops['labels_pl']: argmax_labels,
                         ops['is_training_pl']: is_training,}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            time_count += time.time() - time_start
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val*BATCH_SIZE)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    return time_count, loss_sum / float(total_seen), total_correct / float(total_seen)

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
