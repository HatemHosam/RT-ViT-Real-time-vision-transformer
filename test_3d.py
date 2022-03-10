import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import time

rgb = o3d.io.read_image('lab images/6.jpeg')
d = o3d.io.read_image('results lab/6.png')
old = time.time()
color = o3d.geometry.Image(rgb)
depth = o3d.geometry.Image(d)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth , convert_rgb_to_intensity=False)

# plt.subplot(1, 2, 1)
# plt.title('NYU grayscale image')
# plt.imshow(rgbd.color)
# plt.subplot(1, 2, 2)
# plt.title('NYU depth image')
# plt.imshow(rgbd.depth, cmap='magma')
# plt.show()


pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd,
    o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault))

pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

now= time.time()

o3d.visualization.draw_geometries([pcd])
print(now-old)