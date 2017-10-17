#!/usr/bin/python2
# -*- coding: utf-8 -*-
"""
An ROS-interface or ROS-Node for `voxnet.py`
TODO: (vincent.cheung.mcer@gmail.com)This node is still under construction.
"""
#import tensorflow as tf
import numpy as np
import os
from glob import glob
import random
# Ros related
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

# Voxnet Related
from voxnet import SUOD_label_dictionary

global pc2_pub

def pc2_handler(data):
    """
    Handler for pointcloud2 segments classfication and re-publish pointcloud2 msg with label.
    """
    pass

if __name__ == '__main__':
    try:
        # Init Ros Node
        rospy.init_node('segment_classified',anonymous=True)

        # Init publisher
        global pc2_pub
        pc2_pub = rospy.Publisher('/voxnet/points_l', pc2, queue_size=10)
        
        # Subscriber
        rospy.Subscriber('/velodyne_points',Imu,on_new_imu)


        rospy.spin()
    except KeyboardInterrupt:
        # Handle
        pass
