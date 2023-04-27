import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2
import open3d as o3d
import copy
import time
import subprocess 
from random import seed
from random import randint

# import pyKinectAzure library from folder
sys.path.insert(1, './pyKinectAzure')
import pykinect_azure as pykinect
from pykinect_azure.utils import Open3dVisualizer


def distance_filter(point_cloud, distance_threshold):

    distance = np.linalg.norm(point_cloud, ord=2, axis=1)
    filter_index = distance < distance_threshold

    return point_cloud[filter_index]

def draw_registration_result(source, target, transformation):
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

def preprocess_point_cloud(pcd, voxel_size):
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
    return pcd_down, pcd_fpfh#down sampled point cloud(for visualization&registraion) and fpfh paramters.

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):

    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    source_down, target_down, source_fpfh, target_fpfh,
    o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=distance_threshold))
    return result#transformation matrix

# use optimal transformation obtained from previous ICP
opt_data = np.load('0_candidate.npz')
opt_trans = opt_data['transformation']

folder_name = input("Create a folder and name it for the processed data: ")
# Create folder for each PC 
parent_directory= "/home/nuc/Desktop/kinect_camera/DATA"
path= os.path.join(parent_directory, folder_name)
os.mkdir(path)


folder_path = "/home/nuc/Desktop/kinect_camera/DATA/new_timestamp" #-------------------------change this to raw data's folder name
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        print('processing', filename)
        # Load the .npz file
        data = np.load(file_path)

        # Access the point cloud and color image data
        pt_left = data['pcd_left']
        pt_right = data['pcd_right']
        img_left = data['img_l']
        img_right = data['img_r']

        # print(f'LEFT  PCD: {np.shape(pt_left)}, LEFT  IMG: {np.shape(img_left)}')
        # print(f'RIGHT PCD: {np.shape(pt_right)}, RIGHT IMG: {np.shape(img_right)}')

        # use distance filter
        distance_threshold = 700 
        pt_left = distance_filter(pt_left, distance_threshold)
        pt_right = distance_filter(pt_right, distance_threshold)

        # convert to o3d vector
        pcd_left = o3d.geometry.PointCloud()
        pcd_left.points = o3d.utility.Vector3dVector(pt_left)
        pcd_right = o3d.geometry.PointCloud()
        pcd_right.points = o3d.utility.Vector3dVector(pt_right)

        # # Down sampling for visualization
        # voxel_size = 10
        # left_down, left_fpfh = preprocess_point_cloud(pcd_left, voxel_size)
        # right_down, right_fpfh = preprocess_point_cloud(pcd_right, voxel_size)#fpfh-->feature
        # draw_registration_result(left_down, right_down, opt_trans)

        ###################################################### PLACE COLOR FILTER HERE, BEFORE THE COMBINATION!!!!!!!!!!!!!!!!!!!

        # combine left and right pcd
        pcd_combined = o3d.geometry.PointCloud()
        pcd_combined.points = o3d.utility.Vector3dVector([*pcd_left.points, *pcd_right.points])

        # Display the processed point cloud
        # plt.scatter(pcd_combined.points[:, 0], pcd_combined.points[:, 1], c=pcd_combined.points[:, 2])
        # plt.show()

        # save processed data
        # np.savez(f'./DATA/{folder_name}/data_{i}.npz', pcd=pcd_combined.points, transformation=opt_trans, img_l=img_left)
        np.savez(f'./DATA/{folder_name}/{filename}', pcd=pcd_combined.points, transformation=opt_trans, img_l=img_left)

print('processing complete!')