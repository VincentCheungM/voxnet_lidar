# Voxnet_lidar
An on going TF implementation on Voxnet to deal with LiDAR pointcloud.

## Reference Paper
```bibtex
@inproceedings{Maturana2015VoxNet,
  title={VoxNet: A 3D Convolutional Neural Network for real-time object recognition},
  author={Maturana, Daniel and Scherer, Sebastian},
  booktitle={Ieee/rsj International Conference on Intelligent Robots and Systems},
  pages={922-928},
  year={2015},
}
```

## Dataset
[Sydney Urban Object Dataset,short for SUOD](http://www.acfr.usyd.edu.au/papers/SydneyUrbanObjectsDataset.shtml)


#### Other LiDAR PointCloud Dataset(not yet support though :D):
[Stanford Track Collection](http://cs.stanford.edu/people/teichman/stc/)  
[KITTI Object Recognition](http://www.cvlibs.net/datasets/kitti/eval_object.php)  
[Semantic 3D](http://www.semantic3d.net/view_dbase.php?chl=2) 


## Requirement
1. [python-pcl](https://github.com/strawlab/python-pcl)
2. [Tensorflow](https://github.com/tensorflow/tensorflow)

## Running
```bash
# converting SUOD bin files to pcd and saving centerlized and rotation augmented voxels in `{name}_{rotate_step}.npy`
python read-bin.py
# training and evaluation, checkpoint and log will be saved in `./voxnet/` folder
python voxnet.py
```

## Current Issue
1. Dataset path needs to be modified in `*.py`
2. The training step is really slow, about 44s. It needs to check implementation of Voxnet architecture.
3. Some folder need to be created before running(e.g., lacking path checker and mkdir in the script)
