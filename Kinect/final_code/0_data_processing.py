import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2
import open3d as o3d
import copy
from random import seed
from random import randint
import matplotlib.image

# import pyKinectAzure library from folder
sys.path.insert(1, './pyKinectAzure')
import pykinect_azure as pykinect
from pykinect_azure.utils import Open3dVisualizer


def distance_filter(pcd, distance_threshold):

    # Convert the point cloud to a numpy array
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # select points that satisfy distance requirement
    distance = np.linalg.norm(points, ord=2, axis=1)*1e6
    filter_index = distance < distance_threshold

    print("distance", distance)
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

folder_name = input("Create a folder and name it for the processed data: ")
# Create folder for each PC 
parent_directory= "/home/nuc/Desktop/kinect_camera/DATA"
path= os.path.join(parent_directory, folder_name)
os.mkdir(path)

# distance threshold
distance_threshold = 900 # 900

# color filter parameter
low_color = [140.0/255.0, 140.0/255.0, 140.0/255.0]  # Lower bound of the color range in BGR format--- grey 140
high_color = [255.0/255.0, 255.0/255.0, 255.0/255.0]  # Upper bound of the color range in BGR format---- white

# ---start processing---
folder_path = "/home/nuc/Desktop/kinect_camera/DATA/Black_screen" #-------------------------change this to raw data's folder name
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        print('processing', filename)
        # Load the .npz file
        data = np.load(file_path)

        # Access the point cloud and color image data
        pt_left = data['pcd_l']
        pt_right = data['pcd_r']
        depth_l = data['depth_l']
        depth_r = data['depth_r']
        trans_color_l = data['trans_color_l']
        trans_color_r = data['trans_color_r']
        matplotlib.image.imsave(f'./DATA/{folder_name}/{filename[:-4]}_left.jpg', trans_color_l)
        matplotlib.image.imsave(f'./DATA/{folder_name}/{filename[:-4]}_right.jpg', trans_color_r)

        # read depth and color image
        color_img_left = o3d.io.read_image(f'./DATA/{folder_name}/{filename[:-4]}_left.jpg') # have to use saved image, cannot read from numpy array
        color_img_right = o3d.io.read_image(f'./DATA/{folder_name}/{filename[:-4]}_right.jpg')
        depth_img_left = o3d.geometry.Image(depth_l)            # have to read directly from numpy array, cannot use saved image
        depth_img_right = o3d.geometry.Image(depth_r)

        # generate rgbd image from depth and color image
        depth_scale = 1000
        rgbd_image_left = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img_left, depth_img_left, depth_scale = depth_scale, convert_rgb_to_intensity=False)
        rgbd_image_right = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img_right, depth_img_right, depth_scale = depth_scale, convert_rgb_to_intensity=False)
        
        # generate point cloud from rgbd image
        pcd_left = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_left, left_cam_intrin)
            # rgbd_image_left,
            # o3d.camera.PinholeCameraIntrinsic(
            #     o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        pcd_right = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_right, right_cam_intrin)
            # rgbd_image_right,
            # o3d.camera.PinholeCameraIntrinsic(
            #     o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))


        #---------------------------------------display--------------------------------------------
        # # Visualize the point cloud using the Open3D visualizer
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd_left)
        # vis.run()
        # vis.destroy_window()

        # points = np.asarray(pcd_left.points)
        # colors = np.asarray(pcd_left.colors)
        # print("Is point cloud empty? ",np.all(points == 0))
        # print("Is color info empty? ",np.all(colors == 0))
        # print("points", points.shape)
        # print("colors", colors.shape)
        #---------------------------------------display--------------------------------------------
        
        # # implement distance filter (error)
        # pcd_left = distance_filter(pcd_left, distance_threshold)
        # pcd_right = distance_filter(pcd_right, distance_threshold)

        # distance filter fails due to inconsistance of depth scale(may caused by different camera intrinsics, raw uses Kinect azure, and rgbd uses pinhole). use raw pcd directly from kinect as input, get the index, then apply index to the pcd generated from rgbd image
        left_index = get_distance_index(pt_left, distance_threshold)
        right_index = get_distance_index(pt_right, distance_threshold)

        l_points = np.asarray(pcd_left.points)
        l_colors = np.asarray(pcd_left.colors)
        point_left = l_points[left_index]
        color_left = l_colors[left_index]

        filted_pcd_left = o3d.geometry.PointCloud()
        filted_pcd_left.points = o3d.utility.Vector3dVector(point_left)
        filted_pcd_left.colors = o3d.utility.Vector3dVector(color_left)

        r_points = np.asarray(pcd_right.points)
        r_colors = np.asarray(pcd_right.colors)
        point_right = r_points[right_index]
        color_right = r_colors[right_index]

        filted_pcd_right = o3d.geometry.PointCloud()
        filted_pcd_right.points = o3d.utility.Vector3dVector(point_right)
        filted_pcd_right.colors = o3d.utility.Vector3dVector(color_right)

        
        #---------------------------------------display--------------------------------------------
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
        #---------------------------------------display--------------------------------------------


        # draw_registration_result(pcd_left, pcd_right, opt_trans)# before color filter
        # implement color filter
        # filted_pcd_left = pcd_color_filter(filted_pcd_left, low_color, high_color)
        # filted_pcd_right = pcd_color_filter(filted_pcd_right, low_color, high_color)
        # Check if pcd is empty
        # points = np.asarray(filted_pcd_left.points)
        # colors = np.asarray(filted_pcd_left.colors)
        # print("Is point cloud empty? ",np.all(points == 0))
        # print("points", points.shape)
        # print("colors", colors.shape)

        
        scale_factor_left = 1 / 5.947623985970278e-07 
        scale_factor_right = 1 / 5.958847148072367e-07
        filted_pcd_left.scale(scale_factor_left, center=filted_pcd_left.get_center())
        filted_pcd_right.scale(scale_factor_right, center=filted_pcd_right.get_center())

        # draw_registration_result(filted_pcd_left, filted_pcd_right, opt_trans) # after color filter

    
        #---------------------------------------display--------------------------------------------
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
        #---------------------------------------display--------------------------------------------


        #-------compute the point cloud scale distance 
        # p1 = filted_pcd_left.points[0]
        # p2 = filted_pcd_left.points[1]
        # # Calculate the distance between the two points in the original data
        # dist_orig = np.linalg.norm(p1 - p2)
        # print("dist_orig",dist_orig)


        # current_transformation = np.identity(4)
        # draw_registration_result(filted_pcd_right, filted_pcd_left, current_transformation)
        # Check if pcd is empty
        # points = np.asarray(filted_pcd_left.points)
        # colors = np.asarray(filted_pcd_left.colors)
        # print("Is point cloud empty? ",np.all(points == 0))
        # print("points", points.shape)
        # print("colors", colors.shape)
        #-----------------------------------------------Downsampling
        voxel_size = 10   # 0.05means 5cm for this dataset
        source_down, source_fpfh = preprocess_point_cloud(filted_pcd_left, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(filted_pcd_right, voxel_size)

        # #-----------------------------------------------Global registration
        result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
        # # print(result_ransac)

        # Check if pcd is empty
        # points = np.asarray(source_down.points)
        # colors = np.asarray(source_down.colors)
        # print("Is point cloud empty? ",np.all(points == 0))
        # print("down_points", points.shape)
        # print("down_colors", colors.shape)
        draw_registration_result(source_down, target_down, result_ransac.transformation)
        print("trans after global registration ",result_ransac.transformation)

        #-----------------------------------------------Fast Global registration
        # result_fast = execute_fast_global_registration(source_down, target_down,
        #                                             source_fpfh, target_fpfh,
        #                                             voxel_size)
        # draw_registration_result(source_down, target_down, result_fast.transformation)


        #-----------------------------------------------point to plane ICP
        IPC_init = result_ransac.transformation # current_transformation
        result_icp = o3d.pipelines.registration.registration_icp(
            source_down, target_down, 0.02, IPC_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print("trans after icp", result_icp.transformation)
        draw_registration_result_original_color(source_down, target_down,
                                                result_icp.transformation)
        


        #-----------------------------------------------color ICP
        # voxel_radius = [0.04, 0.02, 0.01]
        # max_iter = [50, 30, 14]
        # print("3. Colored point cloud registration")
        # for scale in range(3):
        #     iter = max_iter[scale]
        #     radius = voxel_radius[scale]
        #     print([iter, radius, scale])

        #     print("3-1. Downsample with a voxel size %.2f" % radius)
        #     source_down = filted_pcd_right.voxel_down_sample(radius)
        #     target_down = filted_pcd_left.voxel_down_sample(radius)

        #     print("3-2. Estimate normal.")
        #     source_down.estimate_normals(
        #         o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        #     target_down.estimate_normals(
        #         o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        #     print("3-3. Applying colored point cloud registration")
        #     result_icp = o3d.pipelines.registration.registration_colored_icp(
        #         source_down, target_down, radius, current_transformation,
        #         o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        #         o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
        #                                                         relative_rmse=1e-6,
        #                                                         max_iteration=iter))
        #     current_transformation = result_icp.transformation
        #     print(result_icp)
        # draw_registration_result_original_color(filted_pcd_right, filted_pcd_left,
        #                                         result_icp.transformation)


        #-----------------------------------------------general ICP

        
        # IPC_init = result_ransac.transformation # current_transformation
        # icp_threshold = 0.05  #5.947623985970278e-08
        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #     source_down, target_down, icp_threshold, IPC_init,
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
        # print("trans after icp", reg_p2p.transformation)
        # o3d.visualization.draw_geometries([filted_pcd_right, filted_pcd_left])
        # draw_registration_result(source_down, target_down, reg_p2p.transformation)


        # # combine left and right pcd
        # pcd_combined = o3d.geometry.PointCloud()
        # pcd_combined.points = o3d.utility.Vector3dVector([*filted_pcd_left.points, *filted_pcd_right.points])
        # pcd_combined.colors = o3d.utility.Vector3dVector([*filted_pcd_left.colors, *filted_pcd_right.colors])

        # # save processed data
        # np.savez(f'./DATA/{folder_name}/{filename}', transformation=opt_trans, pcd_points=pcd_combined.points, pcd_colors=pcd_combined.colors)
        



# test whether to add icp before color filter or after

    
print('processing complete!')