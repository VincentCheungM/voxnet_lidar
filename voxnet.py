 #!/usr/bin/python2
# -*- coding: utf-8 -*-
"""
Voxnet implementation on tensorflow
"""
import tensorflow as tf
import numpy as np
import os
from glob import glob
import random


# label dict for Sydney Urban Object Dataset, ref:http://www.acfr.usyd.edu.au/papers/SydneyUrbanObjectsDataset.shtml
SUOD_label_dictionary = {'4wd': 0, 'building': 1, 'bus': 2, 'car': 3, 'pedestrian': 4, 'pillar': 5, 'pole': 6,
                    'traffic_lights': 7, 'traffic_sign': 8, 'tree': 9, 'truck': 10, 'trunk': 11, 'ute': 12,
                    'van': 13}

# TODO: (vincent.cheung.mcer@gmail.com) combine generator `gen_batch_function` and data collector `get_all_data` into a class
def gen_batch_function(data_folder,batch_size):
    """
    Generate function to create batches of training data.
    
    Args:
    `data_folder`:path to folder that contains all the `npy` datasets.
    
    Ret:
    `get_batches_fn`:generator function(batch_size)
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data.

        Args:
        `batch_size`:Batch Size

        Ret:
        return Batches of training data
        """
        grid_paths = glob(os.path.join(data_folder, 'voxel_npy', '*.npy'))
        # TODO:(vincent.cheung.mcer@gmail.com) not yet add support for multiresolution npy data
        # grid_paths_r2 = glob(os.path.join(data_folder, 'voxel_npy_r2', '*.npy'))
        
        # shuffle data
        random.shuffle(grid_paths)

        for batch_i in range(0, len(grid_paths), batch_size):
            grids = []
            labels = []
            for grid_path in grid_paths[batch_i:batch_i+batch_size]:
                # extract the label from path+file_name: e.g.`./voxel_npy/pillar.2.3582_12.npy`
                file_name = grid_path.split('/')[-1] #`pillar.2.3582_12.npy`
                label = SUOD_label_dictionary[file_name.split('.')[0]] #dict[`pillar`]
                # load *.npy
                grid = np.load(grid_path)
                labels.append(label)
                grids.append(grid)
                
            yield np.array(grids), np.array(labels)
    return get_batches_fn(batch_size)

def get_all_data(data_folder, mode='train', type='dense'):
    """
    Get all voxels and corresponding labels from `data_folder`.

    Args:
    `data_folder`:path to the folder that contains `voxel_npy_train` and `voxel_npy_eval`.
    `mode`:folder that contains all the `npy` datasets 
    `type`:type of npy for future use in sparse tensor, values={`dense`,`sparse`} 

    Ret:
    `grids`:list of voxel grids
    `labels`:list of labels
    """
    sub_path = 'voxel_npy_'+mode
    grid_paths = glob(os.path.join(data_folder, sub_path, '*.npy'))
    
    # TODO:(vincent.cheung.mcer@gmail.com) not yet add support for multiresolution npy data
    # TODO:(vincent.cheung.mcer@gmail.com) not yet support sparse npy
    # grid_paths_r2 = glob(os.path.join(data_folder, 'voxel_npy_r2', '*.npy'))    
    grids=[]
    labels=[]
    for grid_path in grid_paths:
        # extract the label from path+file_name: e.g.`./voxel_npy/pillar.2.3582_12.npy`
        file_name = grid_path.split('/')[-1] #`pillar.2.3582_12.npy`
        label = SUOD_label_dictionary[file_name.split('.')[0]] #dict[`pillar`]
        # load *.npy
        grid = np.load(grid_path).astype(np.float32)
        labels.append(label)
        grids.append(grid)
    return grids, labels

def save_inference_sample():
    """
    # TODO:(vincent.cheung.mcer@gmail.com) to collect voxels and predicted labels
    """
    pass

class Voxnet(object):
    def __init__(self, learning_rate=0.001, num_classes=14, batch_size=32, epochs=64):
        """
        Init paramters
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        # to enable tf logging info
        tf.logging.set_verbosity(tf.logging.INFO)

    def voxnet_fn(self, features, labels, mode):
        """
        Voxnet tensorflow graph.
        It follows description from this TensorFlow tutorial:
        `https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts`
        
        Args:
        `features`:default paramter for tf.model_fn
        `labels`:default paramter for tf.model_fn
        `mode`:default paramter for tf.model_fn

        Ret:
        `EstimatorSpec`:    predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
        """
        input_layer = tf.reshape(features['x'], [-1, 32, 32, 32, 1])
        # Layer 1: 3D conv(filters_num=32, filter_kernel_size=5, strides=2)
        # Input(32*32*32), Output:(14*14*14)*32
        conv1 = tf.layers.conv3d(inputs=input_layer, filters=32, kernel_size=[5,5,5], strides=[2,2,2],name='conv1')
        # Layer 2: 3D conv(filters_num=32, filter_kernel_size=3, strides=1)
        # Max-pooling (2*2*2)
        # Input(32*32*32)*32, Output:(6*6*6)*32
        conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], strides=[1,1,1],name='conv2')
        # TODO: (vincent.cheung.mcer@gmail.com) not sure about the pool_size
        max_pool1 = tf.layers.max_pooling3d(inputs=conv2, pool_size=2,strides=2)
        # TODO: (vincent.cheung.mcer@gmail.com), later can try 3D conv instead of Fully Connect dense layer
        max_pool1_flat = tf.reshape(max_pool1,[-1,6*6*6*32])
        # Layer 3: Fully Connected 128
        # Input (6*6*6)*32, Output:(128)
        dense4 = tf.layers.dense(inputs=max_pool1_flat,units=128)
        # Layer 4: Fully Connected Output
        # Input: (128), Output:K class
        dense5 = tf.layers.dense(inputs=dense4,units=self.num_classes)
        logits = dense5

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            'classes': tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self.num_classes)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)
        tf.summary.scalar("loss_voxel", loss)
        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=labels, predictions=predictions['classes'])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv,data_folder='./',batch_size=8,epochs=8):
    """
    The main function for voxnet training and evaluation.
    """
    voxet = Voxnet()
    # Voxnet Estimator: model init
    voxel_classifier = tf.estimator.Estimator(
        model_fn=voxet.voxnet_fn, model_dir='./voxnet/')

    # Trainning data collector
    grids_list, labels_list = get_all_data(data_folder,mode='train')
    train_data = np.array(grids_list)
    train_labels = np.array(labels_list)
    # Evaluating data collector
    eval_grids_list, eval_labels_list = get_all_data(data_folder,mode='eval')
    eval_data = np.array(eval_grids_list)
    eval_labels = np.array(eval_labels_list)

    print('data get')

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=epochs,
        shuffle=True)

    print ('train start')

    voxel_classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[logging_hook])
    
    print('train done')

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = voxel_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == '__main__':
    # run the main function and model_fn, according to Tensorflow R1.3 API
    tf.app.run(main=main, argv=['./'])