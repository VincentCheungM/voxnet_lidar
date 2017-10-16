# -*- coding: utf-8 -*-
""" Simple example for loading object binary data. """
import numpy as np
import sys
import pcl

names = ['t','intensity','id',
         'x','y','z',
         'azimuth','range','pid']

formats = ['int64', 'uint8', 'uint8',
           'float32', 'float32', 'float32',
           'float32', 'float32', 'int32']


def fold_bin2pcd(fold='./folds/fold0.txt'):
    """
    read bins from `fold` and convert into `*.pcd`  
    Args:
    `fold` the path of fold*.txt that need to convert into pcd
    
    """
    binType = np.dtype( dict(names=names, formats=formats) )
    with open(fold) as f:
        files = f.readlines()
        for file in files:
            file_name =  file.split('\n')[0]
            data = np.fromfile('./objects/'+file_name, binType)
            # 3D points, one per row
            P = np.vstack([ data['x'], data['y'], -data['z'] ]).T
            cloud=pcl.PointCloud(P)
            pcl.save(cloud,'./pcd/{}.pcd'.format(file_name.split('.bin')[0]))
    print('process done with {}'.format(fold))

def points_to_voxel(points,voxel_size=(24,24,24),padding_to_size=(32,32,32),resolution=0.1):
    """
    Convert `points` to centerlized voxel with size `voxel_size` and `resolution`, then padding zero to
    `padding_to_size`. The outside part is cut, rather than scaling the points.  
    Args:
    `points`:pointcloud in 3D numpy.ndarray 
    `voxel_size`:the centerlized voxel size, default (24,24,24) 
    `padding_to_size`:the size after zero-padding, default (32,32,32)   
    `resolution`:the resolution of voxel, in meters     
    Ret:
    `voxel`:32*32*32 voxel occupany grid    
    `inside_box_points`:pointcloud inside voxel grid
    """
    if abs(resolution) < sys.float_info.epsilon:
        print ('error input, resolution should not be zero')
        return None,None
    min_box_coor = (np.min(points[:,0]),np.min(points[:,1]),np.min(points[:,2]))
    # print min_box_coor
    # print points

    # filter outside voxel_box by using passthrough filter
    points[:,0] = points[:,0] - min_box_coor[0]
    points[:,1] = points[:,1] - min_box_coor[1]
    points[:,2] = points[:,2] - min_box_coor[2]

    x_logical = np.logical_and((points[:, 0] < voxel_size[0]*resolution), (points[:, 0] >= 0))
    y_logical = np.logical_and((points[:, 1] < voxel_size[0]*resolution), (points[:, 1] >= 0))
    z_logical = np.logical_and((points[:, 2] < voxel_size[0]*resolution), (points[:, 2] >= 0))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))
    inside_box_points=points[xyz_logical]
    
    voxel = np.zeros(padding_to_size)
    center_points = inside_box_points+(padding_to_size[0]-voxel_size[0])*resolution/2
    # print min_box_coor
    # print points
    # print inside_box_points.shape
    # print inside_box_points
    # print center_points

    voxel[(center_points[:, 0]/resolution).astype(int), (center_points[:, 1]/resolution).astype(int), (center_points[:, 2]/resolution).astype(int)] = 1
    # print np.count_nonzero(voxel)
    # print np.unique((center_points/resolution).astype(int),axis=0).shape
    return voxel,inside_box_points

if __name__=='__main__':
    fold = './folds/fold0.txt'
    #cloud = pcl.PointCloud_PointXYZ()
    if len(sys.argv) >1:
        fold=sys.argv[1]
    binType = np.dtype( dict(names=names, formats=formats) )
    with open(fold) as f:
        success_cnt = 0
        all_cnt = 0
        files = f.readlines()
        for file in files:
            file_name = file.split('\n')[0]
            data = np.fromfile('./objects/'+file_name, binType)
            # 3D points, one per row
            P = np.vstack([ data['x'], data['y'], -data['z'] ]).T
            voxel,scale_points = points_to_voxel(P)
            print ('processsing {} in {}'.format(file_name,fold))
            # print P.shape
            # print voxel.shape,scale_points.shape
            all_cnt+=1
            # save pointcloud from bin
            if P.shape[0]>0:
                cloud=pcl.PointCloud(P)
                # pcl.save(cloud,'./pcd/{}.pcd'.format(file_name.split('.bin')[0]))
                #pcl.save(cloud,'./pcd/{}.pcd'.format(file_name.split('.bin')[0]))
            
            # save filterred pointcloud and voxel
            if scale_points.shape[0]>0:
                success_cnt+=1
                cloud=pcl.PointCloud(scale_points)
                # pcl.save(cloud,'./pcd/{}.pcd'.format(file_name.split('.bin')[0]))
                pcl.save(cloud,'./pcd_voxel1/{}.pcd'.format(file_name.split('.bin')[0]))
            else:
                print ('processed {} is empty'.format(file_name))

    print ('process done :{}/{} success'.format(success_cnt,all_cnt))
