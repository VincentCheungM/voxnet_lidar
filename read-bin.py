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
    Read bins from `fold` as x,y,z and convert into `*.pcd`  

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
    # The min 3d box coordinate of the voxel, same as pcl::voxelGrid::getMinBoxCoordinates
    min_box_coor = (np.min(points[:,0]),np.min(points[:,1]),np.min(points[:,2]))
    # print min_box_coor
    # print points

    # filter outside voxel_box by using passthrough filter
    # set the nearest point as (0,0,0)
    points[:,0] = points[:,0] - min_box_coor[0]
    points[:,1] = points[:,1] - min_box_coor[1]
    points[:,2] = points[:,2] - min_box_coor[2]

    x_logical = np.logical_and((points[:, 0] < voxel_size[0]*resolution), (points[:, 0] >= 0))
    y_logical = np.logical_and((points[:, 1] < voxel_size[0]*resolution), (points[:, 1] >= 0))
    z_logical = np.logical_and((points[:, 2] < voxel_size[0]*resolution), (points[:, 2] >= 0))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))
    inside_box_points=points[xyz_logical]
    
    # Init voxel grid with zero padding_to_size=(32*32*32) and set the occupany grid
    voxel = np.zeros(padding_to_size)
    center_points = inside_box_points+(padding_to_size[0]-voxel_size[0])*resolution/2 #centerlized
    voxel[(center_points[:, 0]/resolution).astype(int), (center_points[:, 1]/resolution).astype(int), (center_points[:, 2]/resolution).astype(int)] = 1
    # print np.count_nonzero(voxel)
    # print np.unique((center_points/resolution).astype(int),axis=0).shape
    # print min_box_coor
    # print points
    # print inside_box_points.shape
    # print inside_box_points
    # print center_points
    return voxel,inside_box_points

def points_self_rotatoin(points,rot_rad):
    """
    TODO: Perform anti-clockwise rotation on `points` with radian `rot_rad` by using rotation matrix,
    according to the middle point of `points`.

    Args:
    `points`:pointcloud in 3D numpy.ndarray
    `rot_rad`:rotation radian

    Ret:
    `rot_points`:rotated points in 3D numpy.ndarray
    """
    # stack points to [x,y,1], size:nx3
    stack_points = np.hstack((points[:,:2],np.ones(points.shape[0],1)))
    # rotation matrix with center in the middle
    rot_mat=np.array([[np.cos(rot_rad),np.sin(rot_rad),0],
	    [-np.sin(rot_rad),np.cos(rot_rad),0],
	    [-mid[0]*np.cos(rot_rad)+mid[1]*np.sin(rot_rad)+mid[0],-mid[0]*np.sin(rot_rad)-mid[1]*np.cos(rot_rad)+mid[1],1]])
    # [x0,y0,1] = [x,y,1] * rot_matrix, size:(n*3) * (3*3) -> n*3
    rot_points = stack_points.dot(rot_mat)
    # repack points from [x0,y0,1] to [x0,y0,z]
    rot_points = np.hstack((rot_points[:,:2],points[:,2].reshape(-1,1)))
    #TODO:(vincent.cheung.mcer@gmail.com) Not yet check
    return rot_points
    

def points_rotation(points,rot_rad):
    """
    Perform anti-clockwise rotation on `points` with radian `rot_rad` by using rotation matrix,
    around the z-axis.

    Args:
    `points`:pointcloud in 3D numpy.ndarray
    `rot_rad`:rotation radian

    Ret:
    `rot_points`:rotated points in 3D numpy.ndarray
    """
    # sub points from [x,y,z] to [x,y]
    sub_points = points[:,:2]
    # rotation matrix with center in (0,0)
    rot_mat=np.array([[np.cos(rot_rad),np.sin(rot_rad)],
	    [-np.sin(rot_rad),np.cos(rot_rad)]])
    # [x0,y0] = [x,y] * rot_matrix, size:(n*2) * (2*2) -> n*2
    rot_points = sub_points.dot(rot_mat)
    # repack points from [x0,y0,1] to [x0,y0,z]
    rot_points = np.hstack((rot_points[:,:2],points[:,2].reshape(-1,1))) 
    return rot_points
    

def data_augmentation(points,voxel_size=(24,24,24),padding_to_size=(32,32,32),resolution=0.1,rot_step=12):
    """
    Pointcloud voxelization, and data augmentation by rotation on z-axis `rot_steps` times.

    Args:
    `points`:pointcloud in 3D numpy.ndarray 
    `voxel_size`:the centerlized voxel size, default (24,24,24) 
    `padding_to_size`:the size after zero-padding, default (32,32,32)   
    `resolution`:the resolution of voxel, in meters
    `rot_step`:rotation steps on z-axis, which means each step will rotate `360/rot_step` degress, defualt 12

    Ret:
    `voxel_list`:list of 32*32*32 voxel occupany grid after voxelization and augmentation    
    `inside_box_points_list`:pointcloud inside voxel grid after voxelization and augmentation
    """
    voxel_list=[]
    inside_box_points_list=[]
    
    for step in range(1, rot_step+1):
        # rotate points
        rot_points = points_rotation(points=points,rot_rad=2*np.pi/step)
        # rotated points voxelization and centerlization
        voxel, inside_box_points = points_to_voxel(points=rot_points)
        voxel_list.append(voxel)
        inside_box_points_list.append(inside_box_points)

    return voxel_list, inside_box_points_list
    

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
            #P = points_rotation(P,np.pi/2).astype(np.float32)
            #voxel,scale_points = points_to_voxel(P)
            steps = 12
            voxel_list, scale_points_list = data_augmentation(points=P,rot_step=steps)
            print ('processsing {} in {}'.format(file_name,fold))
            # print P.shape
            # print voxel.shape,scale_points.shape
            all_cnt+=steps #all_cnt+=1

            # save pointcloud from bin
            # if P.shape[0]>0:
            #     cloud=pcl.PointCloud(P) # P should be type: float
            #     # pcl.save(cloud,'./pcd/{}.pcd'.format(file_name.split('.bin')[0]))
            #     #pcl.save(cloud,'./pcd/{}.pcd'.format(file_name.split('.bin')[0]))
            
            # save filterred pointcloud and voxel
            idx=0
            for voxel, scale_points in zip(voxel_list,scale_points_list):
                idx+=1
                if scale_points.shape[0]>0:
                    success_cnt+=1
                    cloud=pcl.PointCloud(scale_points.astype(np.float32))
                    pcl.save(cloud,'./pcd_voxel2/{}_{}.pcd'.format(file_name.split('.bin')[0],idx))
                    np.save('./voxel_npy/{}_{}.npy'.format(file_name.split('.bin')[0],idx),voxel)
                else:
                    print ('processed {} is empty'.format(file_name))

    print ('process done :{}/{} success, {} failed'.format(success_cnt,all_cnt,all_cnt-success_cnt))
