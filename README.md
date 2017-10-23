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
[Sydney Urban Object Dataset, short for SUOD](http://www.acfr.usyd.edu.au/papers/SydneyUrbanObjectsDataset.shtml)


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

## Experiment
The current model is trained by folder[1-3], and evaluated on folder[4] with resolution `0.2m`, batch size `32` and epoch `8`. And it achieves F1-score at `0.73433015006` for SUOD with only data aumentation rather than other voting technique. The loss is shown as follows:
```CSV
Wall	time	      Step	  Value
1508375747.16754	1	    2.6253740788
1508377549.74742	101	  0.9230182171
1508379332.95241	201	  0.7202908993
1508381070.35217	301	  0.5305011868
1508382793.84819	401	  0.1175880656
1508384521.5706	  501	  0.2003295422
1508386244.20116	601	  0.0610329323
1508387969.68897	701	  0.0604557097
1508389692.15291	801	  0.088219732
1508391414.02475	901	  0.0428960286
1508393137.4318	  1001	0.0754385814
1508394883.46221	1101	0.0614332743
1508396624.6835	  1201	0.0039639813
```

## Current Issue
1. Dataset path needs to be modified in `*.py`
2. The training step is really slow, about 44s. It needs to check implementation of Voxnet architecture.
3. Some folder need to be created before running(e.g., lacking path checker and mkdir in the script)
