import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
data = np.load('_1.npz')

# Access the point cloud and color image data
point_cloud = data['pcd']
color_image = data['img_l']

# Access the transformation matrix
transform = data['transformation']

# Display the color image
plt.imshow(color_image)
plt.show()

# Display the point cloud
plt.scatter(point_cloud[:, 0], point_cloud[:, 1], c=point_cloud[:, 2])
plt.show()
