 #!/usr/bin/python2
# -*- coding: utf-8 -*-
"""
Voxnet implementation on tensorflow
"""
#import tensorflow as tf
import numpy as np
import os
from glob import glob
import random


# label dict for Sydney Urban Object Dataset, ref:http://www.acfr.usyd.edu.au/papers/SydneyUrbanObjectsDataset.shtml
SUOD_label_dictionary = {'4wd': 0, 'building': 1, 'bus': 2, 'car': 3, 'pedestrian': 4, 'pillar': 5, 'pole': 6,
                    'traffic_lights': 7, 'traffic_sign': 8, 'tree': 9, 'truck': 10, 'trunk': 11, 'ute': 12,
                    'van': 13}

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

def save_inference_sample():
    """
    """
    pass

class Voxnet(object):
    def __init__(self):
        """
        Init paramters
        """
        pass
    
    def optimize(self):
        pass
    
    def train_nn(self):
        pass
    
    def run_nn(self):
        pass

    def net(self, x, keep_prob):
        pass

if __name__ == '__main__':
    # test for generator output
    for g,l in gen_batch_function('./',4):
        print g,l