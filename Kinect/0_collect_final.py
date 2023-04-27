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
                                        camera_fps='30FPS',
                                        synchronized_images_only=True, # changed from False to True
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
                                        camera_fps='30FPS',
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
            ret, color_image = capture.get_transformed_color_image()
            if not ret: continue
            # print(f'Capturing COLOR_PCD ...')
            
            return points, color_image

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

    def execute_fast_global_registration(self, source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size): # comment when collecting data

        distance_threshold = voxel_size * 0.5
        print(":: Apply fast global registration with distance threshold %.3f" \
                % distance_threshold)
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
        return result



    def sync_capture(self):

        # configure the sync cameras
        self.sync_capture_config()
        # show number of device
        print(self.device)

        #open3dVisualizer_left = Open3dVisualizer()
        #open3dVisualizer_right = Open3dVisualizer()
        # Inputs 
        folder_name = input("Create a folder and name it for the new test: ")
        reps = input("Enter the number of repetitions: ")

        # Create folder for each PC 
        parent_directory= "/home/nuc/Desktop/kinect_camera/DATA"
        path= os.path.join(parent_directory, folder_name)
        os.mkdir(path)

        # use optimal transformation obtained from previous ICP, instead of running ICP each time
        # opt_data = np.load('0_candidate.npz')
        # opt_trans = opt_data['transformation']

        # Define the IP address and credentials for the Raspberry Pi devices
        pi1 = {'ip': '192.168.1.10', 'username': 'pi', 'password': 'pi'} #192.168.1.10
        pi2 = {'ip': '192.168.1.11', 'username': 'pi', 'password': 'pi'}

        # Define the programs to run on each Raspberry Pi
        program1_0 = 'sudo python3 LED_start.py'
        program1_1 = 'python timestamp_picmaera_test.py' # 100 images   timestamp_picmaera_test.py                  python picamera_test.py 
        program1_2 = 'sudo python3 LED_end.py'

        program2_0 = 'sudo python3 LED_start.py'
        program2_1 = 'python timestamp_picmaera_test.py' # 100 images   timestamp_picmaera_test.py                  python picamera_test.py
        program2_2 = 'sudo python3 LED_end.py'
        # Define the directory where the programs are located
        program_dir = '/home/pi/Collection/'

        # Connect to the Raspberry Pi devices using SSH
        ssh1 = paramiko.SSHClient()
        ssh1.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh1.connect(pi1['ip'], username=pi1['username'], password=pi1['password'])

        ssh2 = paramiko.SSHClient()
        ssh2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh2.connect(pi2['ip'], username=pi2['username'], password=pi2['password'])

        # Change to the program directory on each Raspberry Pi
        stdin1, stdout1, stderr1 = ssh1.exec_command('cd {}; {}'.format(program_dir, program1_0))
        stdin2, stdout2, stderr2 = ssh2.exec_command('cd {}; {}'.format(program_dir, program2_0))
        # Wait for the programs to finish running
        while not stdout1.channel.exit_status_ready() and not stdout2.channel.exit_status_ready():

            time.sleep(1)
            print("Pi_1 turn on LED?", stdout1.channel.exit_status_ready())
            print("Pi_2 turn on LED?", stdout2.channel.exit_status_ready())

        stdin1, stdout1, stderr1 = ssh1.exec_command('cd {}; {}'.format(program_dir, program1_1))
        stdin2, stdout2, stderr2 = ssh2.exec_command('cd {}; {}'.format(program_dir, program2_1))
        # while True:
        start_time = time.time()
        for i in range(int(reps)):
            # power = randint(0,30)

            # capture using the left camera
            pt_left, img_left = self.capture_colorPCD(device_index=0)
            # capture using the right camera
            pt_right, img_right = self.capture_colorPCD(device_index=1)

            # pcd_left = o3d.geometry.PointCloud()
            # pcd_left.points = o3d.utility.Vector3dVector(pt_left)
            # pcd_right = o3d.geometry.PointCloud()
            # pcd_right.points = o3d.utility.Vector3dVector(pt_right)

            # self.draw_registration_result(pcd_left, pcd_right, np.identity(4))

            # voxel_size = 5 # origin: 10
            # left_down, left_fpfh = self.preprocess_point_cloud(pcd_left, voxel_size)
            # right_down, right_fpfh = self.preprocess_point_cloud(pcd_right, voxel_size)

            # print(f'Downsampled PCD: {np.shape(left_down.points)}')


            #--------------dont need this when not using ICP
            # # fast global registration
            # start = time.time()
            # result_fast = self.execute_fast_global_registration(left_down, right_down,
            #                                                     left_fpfh, right_fpfh,
            #                                                     voxel_size)

            # print("Fast global registration took %.3f sec.\n" % (time.time() - start))
            # print(result_fast)
            # print(result_fast.transformation)
            # #self.draw_registration_result(left_down, right_down, result_fast.transformation)


            

            # # ICP
            # print("\n\nApply point-to-point ICP")
            # threshold = 12
            # trans_init = result_fast.transformation
            # start = time.time()
            # reg_p2p = o3d.pipelines.registration.registration_icp(
            #     left_down, right_down, threshold, trans_init,
            #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
            # print("ICP registration took %.3f sec.\n" % (time.time() - start))
            # print(reg_p2p)
            # print(f"Transformation is:\n {reg_p2p.transformation}")
            # pcd_left_crop = pcd_left.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-200, -400, 1]),
			# 												   np.array([0, 200, 320])))
            # pcd_right_crop = pcd_right.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([0, -400, 1]),
			# 												   np.array([80, 200, 500])))





            # original
            # self.draw_registration_result(left_down, right_down, reg_p2p.transformation)
            # self.draw_registration_result(pcd_left_crop, pcd_right_crop, reg_p2p.transformation)

            # optimal trans
            # self.draw_registration_result(left_down, right_down, opt_trans)
            # self.draw_registration_result(pcd_left_crop, pcd_right_crop, opt_trans)


        
            # open3dVisualizer_left(pcd_left.points, img_left)
            # pcd_combined = o3d.geometry.PointCloud()
            # pcd_combined.points = o3d.utility.Vector3dVector([*pcd_left.points, *pcd_right.points])
            # The folder name need to be mandatory as if there isn't a folder name it will return an error
            # np.savez(f'./DATA/{folder_name}/data_{i}.npz', pcd=pcd_combined.points, transformation=opt_trans, img_l=img_left)
            npdt = np.datetime64(datetime.now())
            dt = npdt.astype(datetime)

            timestamp = dt.strftime('%d_%H%M%S_%f')[:-3]
            filename = "data_{}_{}".format(str(i), timestamp)

            np.savez(f'./DATA/{folder_name}/{filename}.npz', pcd_left=pt_left, pcd_right=pt_right, img_l=img_left, img_r=img_right)
        
        end_time = time.time()  # record the end time
        time_spent = end_time - start_time  # calculate the time spent
        print(f"Time spent: {time_spent:.2f} seconds")  # print the time spent


        stdin1, stdout1, stderr1 = ssh1.exec_command('cd {}; {}'.format(program_dir, program1_2))
        stdin2, stdout2, stderr2 = ssh2.exec_command('cd {}; {}'.format(program_dir, program2_2))

        while not stdout1.channel.exit_status_ready() and not stdout2.channel.exit_status_ready():

            time.sleep(1)
            print("Pi_1 turn off LED? ", stdout1.channel.exit_status_ready())
            print("Pi_2 turn off LED? ", stdout2.channel.exit_status_ready())
        # Close the SSH connections
        ssh1.close()
        ssh2.close()



if __name__ == "__main__":

    # change working directory
    os.chdir(sys.path[0])

    # class instance
    kinect = KINECT()
    # kinect.show_colorImage()
    # kinect.show_depthImage()
    # kinect.show_pointCloud()
    # kinect.show_colorPointCloud()
    # kinect.show_depth2Color()
    # kinect.show_color2Depth()
    kinect.sync_capture()
    






# import pykinect_azure as pykinect
# from pykinect_azure.utils import Open3dVisualizer
# import numpy as np
# import open3d as o3d

# # Initialize Kinect Azure cameras
# kinect1 = pykinect.PyKinectRuntime(pykinect.FrameSourceTypes_Color | pykinect.FrameSourceTypes_Depth)
# kinect2 = pykinect.PyKinectRuntime(pykinect.FrameSourceTypes_Color | pykinect.FrameSourceTypes_Depth)

# # Set up Open3D visualizer
# visualizer = Open3dVisualizer()

# # Loop over frames
# while True:
#     # Capture frames from both cameras
#     if kinect1.has_new_color_frame() and kinect1.has_new_depth_frame() and \
#         kinect2.has_new_color_frame() and kinect2.has_new_depth_frame():
#         # Get color and depth data from both cameras
#         color1 = kinect1.get_last_color_frame()
#         depth1 = kinect1.get_last_depth_frame()
#         color2 = kinect2.get_last_color_frame()
#         depth2 = kinect2.get_last_depth_frame()
        
#         # Generate point clouds from both cameras
#         pc1 = generate_point_cloud(color1, depth1)
#         pc2 = generate_point_cloud(color2, depth2)
        
#         # Combine point clouds into a single point cloud
#         pc = combine_point_clouds(pc1, pc2)
        
#         # Visualize point cloud
#         visualizer.update_point_cloud(pc)
        
#     # Check for exit condition
#     if visualizer.window_closed:
#         break
        
# # Release resources
# kinect1.close()
# kinect2.close()
# visualizer.close()
