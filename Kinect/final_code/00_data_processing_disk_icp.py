import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2
import open3d as o3d
import copy
from random import seed
from random import randint
import matplotlib.image
import pyautogui
import time

# import pyKinectAzure library from folder
sys.path.insert(1, './pyKinectAzure')
import pykinect_azure as pykinect
from pykinect_azure.utils import Open3dVisualizer


def distance_filter(pcd, lower_distance_threshold, upper_distance_threshold):

    # Convert the point cloud to a numpy array
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # select points that satisfy distance requirement
    distance = np.linalg.norm(points, ord=2, axis=1)
    # distance = np.linalg.norm(points, ord=2, axis=1)*1e6  # for distance filter using points index
    filter_index = np.where((distance < upper_distance_threshold) & (distance > lower_distance_threshold))

    # print("distance", distance)
    # print("filter_index", filter_index)
    selected_points = points[filter_index]
    selected_colors = colors[filter_index]

    # convert filted pcd back to o3d vector
    pcd_filted = o3d.geometry.PointCloud()
    pcd_filted.points = o3d.utility.Vector3dVector(selected_points)
    pcd_filted.colors = o3d.utility.Vector3dVector(selected_colors)

    return pcd_filted 

def pcd_color_filter(pcd, low_color, high_color):

    # Convert the point cloud to a numpy array
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Select the points that have a color within the desired range
    # Open3D assumes the PointCloud's color values are of float type and in range [0, 1] as stated in the doc.
    mask = np.all((colors >= low_color) & (colors <= high_color), axis=1)
    selected_points = points[mask]
    selected_colors = colors[mask]

    # convert filted pcd back to o3d vector
    pcd_filted = o3d.geometry.PointCloud()
    pcd_filted.points = o3d.utility.Vector3dVector(selected_points)
    pcd_filted.colors = o3d.utility.Vector3dVector(selected_colors)

    return pcd_filted

def get_distance_index(point_cloud, distance_threshold):

    distance = np.linalg.norm(point_cloud, ord=2, axis=1)
    filter_index = distance < distance_threshold

    return filter_index

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target],
                                      zoom=0.5,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])
    
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                    zoom=0.5,
                                    front=[0, -1, -1],
                                    lookat=[0, -500, 0],
                                    up=[0, -1, 0])

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
    return pcd_down, pcd_fpfh #down sampled point cloud(for visualization&registraion) and fpfh paramters.

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching( #
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud],
                                    zoom=1,
                                    front=[0, -1, -1],
                                    lookat=[0, -500, 0],
                                    up=[0, -1, 0])
    return inlier_cloud



# Kinect Azure intrinsic parameters obtained from example2dto3d

left_cam_intrin = o3d.open3d.camera.PinholeCameraIntrinsic(512, 512, fx=252.346603, fy=252.433380, cx=251.751358, cy=251.999694)
# left_cam_intrin = np.array([[252.346603, 0, 251.751358],
#         	[0, 252.433380, 251.999694],
#         	[0, 0, 1]])

right_cam_intrin = o3d.open3d.camera.PinholeCameraIntrinsic(512, 512, fx=251.871322, fy=251.900039, cx=253.403625, cy=254.246459)
# right_cam_intrin = np.array([[251.871322, 0, 253.403625],
#         	[0, 251.900039, 254.246459],
#         	[0, 0, 1]])	

# use optimal transformation obtained from previous ICP
opt_data = np.load('0_best_matrix.npz')
opt_trans = opt_data['transformation']




ax = 45.7  #45.3 #45.7
ay = 0
az = 0
idt = np.array([          [-1,     0,          0,     0],
                               [0,      1,          0,      0],
                               [0,      0,          1,      0],
                               [0,      0,          0,      1]])

# back up idt
# idt = np.array([          [-1,     0,          0,     0],
#                                [0,      1,          0,      0],
#                                [0,      0,          1,      0],
#                                [0,      0,          0,      1]])

p_minus = np.array([               [1,      0,          0,     -25],
                               [0,      1,          0,      -800], #-1050
                               [0,      0,          1,      690], #900
                               [0,      0,          0,      1]])


p_plus = np.array([               [1,      0,          0,     -10],
                               [0,      1,          0,      -800], #-1050
                               [0,      0,          1,      690], #900
                               [0,      0,          0,      1]])

# p = np.array([               [1,      0,          0,     8],
#                                [0,      1,          0,      -745],
#                                [0,      0,          1,      950],
#                                [0,      0,          0,      1]])

# idt = np.array([               [-1,      0,          0,     -10],
#                                [0,      1,          0,      -60],
#                                [0,      0,          1,      75],
#                                [0,      0,          0,      1]])

x = np.array([                 [1,      0,          0,      0],
                               [0, np.cos(ax),  np.sin(ax), 0],
                               [0, np.sin(ax), -np.cos(ax), 0],
                               [0,      0,          0,      1]])

y= np.array([                  [np.cos(ay),  0, np.sin(ay), 0],
                               [0,           1,      0,     0],
                               [-np.sin(ay), 0, np.cos(ay), 0],
                               [0,           0,      0,     1]])

z = np.array([                  [np.cos(az), -np.sin(az), 0,    0],
                                [np.sin(az), np.cos(az),  0,    0],
                               [0,              0,        1,    0],
                               [0,              0,        0,    1]])


# opt_transformation = np.matmul(np.matmul(np.matmul(p_minus,x),y),z)






inverse = np.array([[ 9.99714424e-01,  2.34313354e-02,  4.69505918e-03, -12.41256908e+01],
                    [-2.33684157e-02,  1.27 ,         -1.30356508e-02,  5.00409961e+01],
                    [-4.99882073e-03,  1.29222120e-02,  1,           -5.11079101e+01],
                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
                                            # | R11 R12 R13 Tx |
                                            # | R21 R22 R23 Ty |
                                            # | R31 R32 R33 Tz |
                                            # |  0   0   0   1 |
# np.array([[ 9.99714424e-01,  2.34313354e-02,  4.69505918e-03, -3.41256908e+01],
#                     [-2.33684157e-02,  9.99641930e-01, -1.30356508e-02,  1.00409961e+01],
#                     [-4.99882073e-03,  1.29222120e-02,  9.99904010e-01, -2.01079101e+01],
#                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

folder_name = input("Create a folder and name it for the processed data: ")
# Create folder for each PC 
parent_directory= "/home/nuc/Desktop/kinect_camera/DATA"
path= os.path.join(parent_directory, folder_name)
os.mkdir(path)

# distance threshold
left_lower_distance_threshold =1.3 # 1.34 1.31
left_upper_distance_threshold = 10 # 2.6 2.3

right_lower_distance_threshold = 1.3 # 1.32
right_upper_distance_threshold = 10 # 2.6

# color filter parameter
low_color = [200.0/255.0, 200.0/255.0, 200.0/255.0]  # Lower bound of the color range in BGR format--- grey 140
high_color = [255.0/255.0, 255.0/255.0, 255.0/255.0]  # Upper bound of the color range in BGR format---- white

# ---start processing---
folder_path = "/home/nuc/Desktop/kinect_camera/DATA/temp" #-------------------------change this to raw data's folder name    #  Black_screen low_light calibration_cube
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        print('processing', filename)
        # Load the .npz file
        data = np.load(file_path)

        # Access the point cloud and color image data
        pt_left = data['pcd_l']
        pt_right = data['pcd_r']
        # depth_l = data['depth_l']
        # depth_r = data['depth_r']
        # trans_color_l = data['trans_color_l']
        # trans_color_r = data['trans_color_r']

        # matplotlib.image.imsave(f'./DATA/{folder_name}/{filename[:-4]}_left.jpg', trans_color_l)
        # matplotlib.image.imsave(f'./DATA/{folder_name}/{filename[:-4]}_right.jpg', trans_color_r)
        # cv2.imwrite(f'./DATA/{folder_name}/{filename[:-4]}_left.jpg', trans_color_l)
        # cv2.imwrite(f'./DATA/{folder_name}/{filename[:-4]}_right.jpg', trans_color_r)
 
        # read depth and color image
        # color_img_left = o3d.io.read_image(f'./DATA/trans_img/{filename[:-4]}_left.jpg') # can only use saved image, cannot read from numpy array
        # color_img_right = o3d.io.read_image(f'./DATA/trans_img/{filename[:-4]}_right.jpg')

        # color_img_left = o3d.geometry.Image(trans_color_l)            
        # color_img_right = o3d.geometry.Image(trans_color_r)
        # depth_img_left = o3d.geometry.Image(depth_l)            # have to read directly from numpy array, cannot use saved image           03_220240_010_left&right
        # depth_img_right = o3d.geometry.Image(depth_r)

        pt_l = o3d.geometry.PointCloud()
        pt_l.points = o3d.utility.Vector3dVector(pt_left)
        pt_r = o3d.geometry.PointCloud()
        pt_r.points = o3d.utility.Vector3dVector(pt_right)

        # depth_scale = 0.5 # if change this, scale below also needs to be changed        ##unit is milimiter, 0.05 means 5mm
        # rgbd_image_left = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #     color_img_left, depth_img_left, depth_scale = depth_scale, convert_rgb_to_intensity=False)
        # rgbd_image_right = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #     color_img_right, depth_img_right, depth_scale = depth_scale, convert_rgb_to_intensity=False)
        
        # # generate point cloud from rgbd image
        # pcd_left = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_left, left_cam_intrin)
        # pcd_right = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_right, right_cam_intrin)

        #---------------------------------------display--------------------------------------------
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd_left) #pcd_left
        # vis.run()
        # vis.destroy_window()

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd_right) #pcd_right
        # vis.run()
        # vis.destroy_window()

        # points = np.asarray(pcd_left.points)
        # colors = np.asarray(pcd_left.colors)
        # print("Is point cloud empty? ",np.all(points == 0))
        # print("Is color info empty? ",np.all(colors == 0))
        # print("points", points.shape)
        # print("colors", colors.shape)
        # draw_registration_result_original_color(pt_l, pt_r, opt_transformation) # opt_transformation  np.identity(4)
        # draw_registration_result_original_color(pcd_left, pcd_right, opt_transformation) # opt_transformation  np.identity(4)
        #---------------------------------------implement distance filter-------------------------------------------

        # filted_pcd_left = distance_filter(pcd_left, left_lower_distance_threshold, left_upper_distance_threshold)
        # filted_pcd_right = distance_filter(pcd_right, right_lower_distance_threshold, right_upper_distance_threshold)

 
        

        # Visualize the original and down-sampled point clouds side by side
        # o3d.visualization.draw_geometries([pcd_left, down_pcd])


        # #---------------------------------------display--------------------------------------------
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(filted_pcd_left)
        # vis.run()
        # vis.destroy_window()

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(filted_pcd_right)
        # vis.run()
        # vis.destroy_window()

        # draw_registration_result_original_color(filted_pcd_left, filted_pcd_right, opt_transformation)


        #------------------------------compute the point cloud scale distance------------------------------------ 
        # p1 = pcd_left.points[2000]
        # p2 = pcd_right.points[2001]
        # # Calculate the distance between the two points in the original data
        # dist_orig = np.linalg.norm(p1 - p2)
        # print("dist_orig",dist_orig)
        # scale_factor_left = 1 / 0.0033178980094215626 # dist_orig     
        # scale_factor_right = 1 / 0.0033178980094215626
        # pcd_left.scale(scale_factor_left, center=pcd_left.get_center())
        # pcd_right.scale(scale_factor_right, center=pcd_right.get_center())
        # l_center=pcd_left.get_center()
        # r_center=pcd_right.get_center()
        # print("l",l_center)
        # print("r",r_center)
        # draw_registration_result(pcd_left, pcd_right, np.identity(4)) # after color filter

        # left center: [ 0.01944904 -0.06647296  0.68755252]
        # right center: [ 0.03180043 -0.0815349   0.69882019]

        # pcd_left_crop_for_ICP = pt_l.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-150, -70, -40]),  # min np.array([-550, -720, 250])   ### y is hight, change x and z only, 
        #                                                                         np.array([125, 30, 100])))       # max np.array([550, 300, 920])   center:[ 25.23762512 -96.48274231 342.63977432]
        
        # pcd_right_crop_for_ICP = pt_r.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-130, -40, -40]), # min np.array([-500, -520, 250])
        #                                                                         np.array([125, 40, 100])))      # max np.array([500, 300, 1250])   center:[  -9.85006714 -176.27865601  454.47079468]
        
        # pcd_left_crop_for_visual = pcd_left.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-150, -70, -55]),  # min np.array([-550, -720, 250])   ### y is hight, change x and z only, 
        #                                                                         np.array([125, 30, 100])))       # max np.array([550, 300, 920])   center:[ 25.23762512 -96.48274231 342.63977432]
        
        # pcd_right_crop_for_visual = pcd_right.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-130, -40, -40]), # min np.array([-500, -520, 250])
        #                                                                         np.array([125, 40, 100])))      # max np.array([500, 300, 1250])   center:[  -9.85006714 -176.27865601  454.47079468]

        pcd_left_crop_for_ICP = pt_l.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-500, -520, 480]),  # min np.array([-500, -520, 480])
                                                                                np.array([500, 200, 800])))       # max np.array([550, 300, 800])
        
        pcd_right_crop_for_ICP = pt_r.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-500, -520, 480]), # min np.array([-500, -520, 480])
                                                                                    np.array([500, 200, 800])))      # max np.array([500, 300, 800])
        
        # backup
        # pcd_left_crop_for_ICP = pt_l.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-500, -520, 480]),  # min np.array([-500, -520, 480])
        #                                                                         np.array([500, 300, 800])))       # max np.array([550, 300, 800])
        # pcd_right_crop_for_ICP = pt_r.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-500, -520, 480]), # min np.array([-500, -520, 480])
        #                                                                             np.array([500, 300, 800])))      # max np.array([500, 300, 800])
        
        # draw_registration_result(pcd_right_crop_for_visual, pcd_left_crop_for_visual, opt_transformation) ### right, left, mismatch if order changed


        #---------------------------------------display--------------------------------------------
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd_left_crop_for_ICP)
        # vis.run()
        # vis.destroy_window()

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd_right_crop_for_ICP)
        # vis.run()
        # vis.destroy_window()

        # draw_registration_result(pcd_right_crop_for_ICP, pcd_left_crop_for_ICP, idt)


        #---------------------------------------display--------------------------------------------
        # current_transformation = np.identity(4)
        # draw_registration_result(pcd_right_crop_for_ICP, pcd_left_crop_for_ICP, opt_transformation)
        # Check if pcd is empty
        # points = np.asarray(filted_pcd_left.points)
        # colors = np.asarray(filted_pcd_left.colors)
        # print("Is point cloud empty? ",np.all(points == 0))
        # print("points", points.shape)
        # print("colors", colors.shape)

        #-----------------------------------------------Downsampling----------------------------------------
        voxel_size = 10   #10
        source_down, source_fpfh = preprocess_point_cloud(pcd_left_crop_for_ICP, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(pcd_right_crop_for_ICP, voxel_size)

        # #-----------------------------------------------Global registration----------------------------------------
        # result_ransac = execute_global_registration(source_down, target_down,  # with downsampling
        #                                     source_fpfh, target_fpfh,
        #                                     voxel_size)
        # result_ransac = execute_global_registration(source_down, target_down,
        #                                     source_fpfh, target_fpfh,
        #                                     voxel_size)
        # # print(result_ransac)

        # Check if pcd is empty
        # points = np.asarray(source_down.points)
        # colors = np.asarray(source_down.colors)
        # print("Is point cloud empty? ",np.all(points == 0))
        # print("down_points", points.shape)
        # print("down_colors", colors.shape)
        # draw_registration_result_original_color(source_down, target_down, result_ransac.transformation)  # 
        # print("trans after global registration ",result_ransac.transformation)

        #-----------------------------------------------Fast Global registration
        # result_fast = execute_fast_global_registration(source_down, target_down,
        #                                             source_fpfh, target_fpfh,
        #                                             voxel_size)
        # draw_registration_result(source_down, target_down, result_fast.transformation)


        #-----------------------------------------------point to plane ICP
        # IPC_init = result_ransac.transformation # current_transformation
        # result_icp = o3d.pipelines.registration.registration_icp(
        #     source_down, target_down, 0.02, IPC_init,
        #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
        # print("trans after icp", result_icp.transformation)
        # draw_registration_result_original_color(source_down, target_down,
        #                                         result_icp.transformation)
        

        #-----------------------------------------------general ICP
        
        # current_transformation = np.identity(4)
        # IPC_init = idt # current_transformation   # result_ransac.transformation
        # icp_threshold = 400  # 1 makes difference max_correspondence_distance
        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #     source_down, target_down, icp_threshold, IPC_init,
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        #     o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-9,
        #                                                     relative_rmse=1e-9,
        #                                                     max_iteration=100000))
        
        IPC_init = idt # backup
        icp_threshold = 400  
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_down, target_down, icp_threshold, IPC_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-8,
                                                            relative_rmse=1e-8,
                                                            max_iteration=50000))
        
        # draw_registration_result_original_color(source_down, target_down, reg_p2p.transformation)
        # print("trans before icp", opt_transformation)
        # print("trans after icp", reg_p2p.transformation)
        # trans = np.matmul(np.matmul(np.matmul(np.matmul(p_minus,x),y),z),reg_p2p.transformation)
        # trans = np.matmul(np.matmul(np.matmul(np.matmul(p_plus,x),y),z),reg_p2p.transformation)
        # draw_registration_result_original_color(source_down, target_down, trans)
        # print("before ",reg_p2p.transformation)
        reg_p2p.transformation = np.matmul(np.matmul(np.matmul(np.matmul(p_plus,x),y),z),reg_p2p.transformation)

        # print("after ",reg_p2p.transformation)
        # print("trans ",trans)
        # Save the ICP registration result
        
        draw_registration_result_original_color(source_down, target_down, reg_p2p.transformation)
        
        transformed_source = source_down.transform(reg_p2p.transformation)
        newpointcloud = transformed_source + target_down
        

 

        #---------------------------------------ransac plane remove-------------------------------------------
        iter = 18
        pcd = newpointcloud
        for i in range(iter):
            
            points = np.asarray(pcd.points)
            # Planar segmentation
            plane_model, inliers = pcd.segment_plane(distance_threshold=2, ransac_n=3, num_iterations=1000)

            # Get indices of inliers 
            inlier_indices = np.asarray(inliers)
            all_indices = np.arange(len(pcd.points))
            outlier_indices = np.delete(all_indices, inlier_indices)
            new_pcd = pcd.select_by_index(outlier_indices)

            # Separate inliers and outliers
            inlier_cloud = pcd.select_by_index(inlier_indices)
            # Visualize the results
            inlier_cloud.paint_uniform_color([1, 0, 0])
            new_pcd.paint_uniform_color([0, 1, 0])
            # o3d.visualization.draw_geometries([inlier_cloud])
            # o3d.visualization.draw_geometries([new_pcd])
            o3d.visualization.draw_geometries([inlier_cloud, new_pcd],zoom=0.5,
                                    front=[0, -1, -1],
                                    lookat=[0, -500, 0],
                                    up=[0, -1, 0])
            pcd = new_pcd
        # o3d.visualization.draw_geometries([pcd])


        print("Statistical oulier removal")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=300,      # nb_neighbors, which specifies how many neighbors are taken into account in order to calculate the average distance for a given point.
                                                    std_ratio=0.005)                # std_ratio, which allows setting the threshold level based on the standard deviation of the average distances across the point cloud. 
        #                                                                          # The lower this number the more aggressive the filter will be.
        pcd = display_inlier_outlier(pcd, ind)
        o3d.io.write_point_cloud(f'./DATA/{folder_name}/{filename[:-4]}.ply', pcd)

        # print("Radius oulier removal")
        # cl, ind = pcd.remove_radius_outlier(nb_points=250, radius=85)   #250,100     # nb_points, which lets you pick the minimum amount of points that the sphere should contain.
        #                                                                     # radius, which defines the radius of the sphere that will be used for counting the neighbors.
        # pcd = display_inlier_outlier(pcd, ind)

        # o3d.visualization.draw_geometries([pcd],
        #                             zoom=1,
        #                             front=[0, -1, -1],
        #                             lookat=[0, -500, 0],
        #                             up=[0, -1, 0])
        
        

        o3d.visualization.draw_geometries([pcd],
                                        zoom=0.5,
                                        front=[0, -1, -1],
                                        lookat=[0, -500, 0],
                                        up=[0, -1, 0])
        # o3d.visualization.draw_geometries([pcd])

        # screen_width, screen_height = pyautogui.size()
        # screenshot = pyautogui.screenshot()
        # screenshot.save(f'./DATA/{folder_name}/{filename[:-4]}_visual.jpg')



        # #---------------------------------------ransac plane remove-------------------------------------------

        # pcd = new_pcd
        # points = np.asarray(pcd.points)
        # # Planar segmentation
        # plane_model, inliers = pcd.segment_plane(distance_threshold=10, ransac_n=3, num_iterations=1000)

        # # Get indices of inliers 
        # inlier_indices = np.asarray(inliers)
        # all_indices = np.arange(len(pcd.points))
        # outlier_indices = np.delete(all_indices, inlier_indices)
        # new_pcd = pcd.select_by_index(outlier_indices)

        # # Separate inliers and outliers
        # inlier_cloud = pcd.select_by_index(inlier_indices)
        # # Visualize the results
        # inlier_cloud.paint_uniform_color([1, 0, 0])
        # new_pcd.paint_uniform_color([0, 1, 0])
        # o3d.visualization.draw_geometries([inlier_cloud])
        # o3d.visualization.draw_geometries([new_pcd])
        # o3d.visualization.draw_geometries([inlier_cloud, new_pcd])
        # #---------------------------------------ransac plane remove-------------------------------------------

        # pcd = new_pcd
        # points = np.asarray(pcd.points)
        # # Planar segmentation
        # plane_model, inliers = pcd.segment_plane(distance_threshold=5, ransac_n=3, num_iterations=1000)

        # # Get indices of inliers 
        # inlier_indices = np.asarray(inliers)
        # all_indices = np.arange(len(pcd.points))
        # outlier_indices = np.delete(all_indices, inlier_indices)
        # new_pcd = pcd.select_by_index(outlier_indices)

        # # Separate inliers and outliers
        # inlier_cloud = pcd.select_by_index(inlier_indices)
        # # Visualize the results
        # inlier_cloud.paint_uniform_color([1, 0, 0])
        # new_pcd.paint_uniform_color([0, 1, 0])
        # o3d.visualization.draw_geometries([inlier_cloud])
        # o3d.visualization.draw_geometries([new_pcd])
        # o3d.visualization.draw_geometries([inlier_cloud, new_pcd])

        # pcd = newpointcloud
        # points = np.asarray(pcd.points)
        # print("points.shape",points.shape)
        # # Planar segmentation
        # plane_model, subset_indices = pcd.segment_plane(distance_threshold=5, ransac_n=3, num_iterations=5000)

        # # Get the complement of the subset indices
        # all_indices = np.arange(len(pcd.points))
        # complement_indices = np.delete(all_indices, subset_indices)
        # new_pcd = pcd.select_by_index(complement_indices)
        # print("complement_indices.shape",complement_indices.shape)
        # # Select the complement of the subset and set it as the new point cloud

        # # Visualize the new point cloud
        # o3d.visualization.draw_geometries([new_pcd])





        # # Extract the largest plane
        # inliers_plane, coefficients_plane = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        
        # max_plane_size = 0
        # for i in range(len(coefficients_plane)):
        #     plane_size = len(inliers_plane[i]) # single element, why len?
        #     if plane_size > max_plane_size:
        #         max_plane_size = plane_size
        #         max_plane_index = i
        # plane_inliers = inliers_plane[max_plane_index]

        # # Remove the plane from the point cloud
        # plane_points = points[plane_inliers]
        # distances = np.abs(np.dot(plane_points, normal) + distance)
        # not_plane_points = points[distances > 0.01]

        # # Create a new point cloud without the plane
        # pcd_plane_removed = o3d.geometry.PointCloud()
        # pcd_plane_removed.points = o3d.utility.Vector3dVector(not_plane_points)

        # # Visualize the results
        # o3d.visualization.draw_geometries([pcd_plane_removed])





























        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        
        # vis.add_geometry(transformed_source)
        # vis.add_geometry(target_down)
        # vis.add_geometry(newpointcloud)

        # view_control = vis.get_view_control()
        # view_control.set_lookat([0, -500, 0])  # Look at the origin
        # view_control.set_up([0, -1, 0])    # Set the up direction to be downwards
        # view_control.set_front([0, -1, -1]) # Set the front direction to be leftwards

        # vis.run()
        # screen_width, screen_height = pyautogui.size()
        # screenshot = pyautogui.screenshot()
        # screenshot.save(f'./DATA/{folder_name}/{filename[:-4]}_visual.jpg')
        # vis.destroy_window()



        # # # save processed data
        np.savez(f'./DATA/{folder_name}/{filename}', transformation=reg_p2p.transformation, source_pcd=source_down.points, target_pcd=target_down.points)
        # np.savez(f'./DATA/{folder_name}/{filename}', transformation=opt_trans, pcd_left_point=filted_pcd_left.points, pcd_left_color=filted_pcd_left.colors, pcd_right_point=filted_pcd_right.points, pcd_right_color=filted_pcd_right.colors)
        



# test whether to add icp before color filter or after

    
print('processing complete!')