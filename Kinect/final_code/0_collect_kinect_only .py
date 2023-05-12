# this code collect original point cloud directly from Kinect Azure and save the data with a timestamp
# the pcd data collection will be started right after the raspberry pi data collection script.
# need to specify the iteration on both raspi and pc

import os, sys
import cv2
import numpy as np
import open3d as o3d
import copy
import time
import subprocess 
from random import seed
from random import randint
import paramiko
from datetime import datetime
import matplotlib.pyplot as plt
# import pyKinectAzure library from folder
sys.path.insert(1, './pyKinectAzure')
import pykinect_azure as pykinect
from pykinect_azure.utils import Open3dVisualizer


class KINECT():

    def __init__(self):
        # initialize the library
        pykinect.initialize_libraries()
        # load camera configuration
        self.device_config = pykinect.default_configuration
        # container for kinect device
        self.device = []
        
    def set_camera_configuration(self, device_index=0,
                                        color_format='JPEG',
                                        color_resolution='720',
                                        depth_mode='WFOV',
                                        camera_fps='30FPS',
                                        synchronized_images_only=False,
                                        depth_delay_off_color_usec=0,
                                        sync_mode='Standalone',
                                        subordinate_delay_off_master_usec=0):
        # set color format
        if color_format == 'JPEG':
            self.device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_MJPG
        elif color_format == 'BGRA':
            self.device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        else:
            print(f'[ERROR] Unknown Color Format: {color_format}')
            exit(-1)

        # set color resolution to 1080P
        if color_resolution == '720':
            self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        elif color_resolution == '1080':
            self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        elif color_resolution == 'OFF':
            self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
        else:
            print(f'[ERROR] Unknown Color Resolution: {color_resolution}')
            exit(-1)
        
        # set depth mode
        if depth_mode == 'NFOV':
            self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
        elif depth_mode == 'WFOV':
            self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        elif depth_mode == 'OFF':
            self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_OFF
        else:
            print(f'[ERROR] Unknown Depth Mode: {depth_mode}')
            exit(-1)

        # camera fps
        if camera_fps == '5FPS':
            self.device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_5
        elif camera_fps == '15FPS':
            self.device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
        elif camera_fps == '30FPS':
            self.device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30

        # synchronized_images_only
        self.device_config.synchronized_images_only = synchronized_images_only

        # depth_delay_off_color_usec
        self.device_config.depth_delay_off_color_usec = depth_delay_off_color_usec

        # wire sync mode
        if sync_mode == 'Standalone':
            self.device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_STANDALONE
        elif sync_mode == 'Master':
            self.device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_MASTER #While in master mode the color camera must be enabled as part of the multi device sync signalling logic. Even if the color image is not needed, the color camera must be running.
        if sync_mode == 'Subordinate':
            self.device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_SUBORDINATE #The 'Sync In' jack is used for synchronization and 'Sync Out' is driven for the next device in the chain.


        # subordinate_delay_off_master_usec
        self.device_config.subordinate_delay_off_master_usec = subordinate_delay_off_master_usec

        print(self.device_config)
        # start the device
   
        self.device.append(pykinect.start_device(device_index=device_index, config=self.device_config))

    def sync_capture_config(self):
        # configure the left camera, NOTICE that left camera is on the right-hand side!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.set_camera_configuration(device_index=0,
                                        color_format='BGRA',
                                        color_resolution='720',
                                        depth_mode='WFOV', 
                                        camera_fps='15FPS', # 0, 5, 15, 30
                                        synchronized_images_only=True, # Setting this to false ensures that the caller receives all of the images received from the camera, regardless of whether the corresponding images expected in the capture are available.
                                        depth_delay_off_color_usec=0,
                                        sync_mode='Master', # Standalone
                                        subordinate_delay_off_master_usec=0)    # The external synchronization timing.
                                        # If this camera is a subordinate, this sets the capture delay between the color camera capture and the external input pulse. A setting of zero indicates that the master and subordinate color images should be aligned.
                                        # This setting does not effect the 'Sync out' connection.
                                        # This value must be positive and range from zero to one capture period.
                                        # If this is not a subordinate, then this value is ignored.
        
                                        


        # configure the right camera
        self.set_camera_configuration(device_index=1,
                                        color_format='BGRA',
                                        color_resolution='720',
                                        depth_mode='WFOV', 
                                        camera_fps='15FPS', # 0, 5, 15, 30
                                        synchronized_images_only=True, # False 
                                        depth_delay_off_color_usec=0,
                                        sync_mode='Subordinate',
                                        subordinate_delay_off_master_usec=1) #---------------------------for sync, change from 0 to 1
    
    def capture_colorPCD(self, device_index):
        print('Capturing colorPCD...')
        while True:
            # perform capture
            capture = self.device[device_index].update()
            # capture point cloud
            ret, points = capture.get_pointcloud()
            if not ret: continue
            ret, color_image = capture.get_color_image() # original color image
            if not ret: continue
            ret, depth_image = capture.get_colored_depth_image() # original depth image   (512, 512, 3) 
            if not ret: continue
            # print(type(depth_image))
            ret, trans_depth_image = capture.get_transformed_colored_depth_image() # depth image in color camera view
            if not ret: continue
            ret, trans_color_image = capture.get_transformed_color_image() # color image in depth camera view  (512, 512, 4) 
            if not ret: continue
            # ret, depth_to_pcd = capture.get_pointcloud_object() # k4a attribute, depth to pcd
            # if not ret: continue
            # print(f'Capturing COLOR_PCD ...')
            
            return points, depth_image, trans_color_image # points, 


    def distance_filter(self, point_cloud, distance_threshold):

        distance = np.linalg.norm(point_cloud, ord=2, axis=1)
        filter_index = distance < distance_threshold

        return point_cloud[filter_index]

    def draw_registration_result(self, source, target, transformation): # comment when collecting data
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                        zoom=0.4559,
                                        front=[0.6452, -0.3036, -0.7011],
                                        lookat=[1.9892, 2.0208, 1.8945],
                                        up=[-0.2779, -0.9482, 0.1556])

    def preprocess_point_cloud(self, pcd, voxel_size): # comment when collecting data
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def sync_capture(self):

        # configure the sync cameras
        self.sync_capture_config()
        # show number of device
        print(self.device)

        # Inputs 
        folder_name = input("Create a folder and name it for the new test: ")
        reps = input("Enter the number of repetitions: ")

        # Create folder for each PC 
        parent_directory= "/home/nuc/Desktop/kinect_camera/DATA"
        path= os.path.join(parent_directory, folder_name)
        os.mkdir(path)






        start_time = time.time()
        for i in range(int(reps)):

            # capture using the left camera
            pt_left,  depth_left, trans_color_left = self.capture_colorPCD(device_index=0) # pt_left, 
            pt_right, depth_right, trans_color_right = self.capture_colorPCD(device_index=1) # pt_right, 

            # while True:
            #     cv2.namedWindow('Transformed color Image',cv2.WINDOW_NORMAL)
            #     cv2.imshow('Transformed color Image',img_left) # img_left[:,:,:3]
            #     if cv2.waitKey(1) == ord('q'): 
            #         break
            # print(img_left)


            # make timestamp 
            npdt = np.datetime64(datetime.now())
            dt = npdt.astype(datetime)
            timestamp = dt.strftime('%d_%H%M%S_%f')[:-3]
            filename = "{}".format(timestamp)

            # save raw data
            np.savez(f'./DATA/{folder_name}/{filename}.npz', pcd_l=pt_left, pcd_r=pt_right, 
                                                            depth_l=depth_left, depth_r=depth_right, 
                                                            trans_color_l=trans_color_left, trans_color_r=trans_color_right, 
                                                            ) 
            # pcd_l=pt_left, pcd_r=pt_right, trans_depth_l=trans_depth_left, trans_depth_r=trans_depth_right, color_l = color_left, color_r = color_right
                                                            
        
        end_time = time.time()  # record the end time
        time_spent = end_time - start_time  # calculate the time spent
        print(f"Time spent: {time_spent:.2f} seconds")  # print the time spent




if __name__ == "__main__":

    # change working directory
    os.chdir(sys.path[0])

    # class instance
    kinect = KINECT()

    kinect.sync_capture()
    