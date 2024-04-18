import open3d as o3d
import numpy as np
import time

print(o3d.__version__)
print("Load a ply point cloud, print it, and render it")
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(ply_point_cloud.path)

# Create a visualization object
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the point cloud to the visualization object
vis.add_geometry(pcd)

# Get the view control
ctr = vis.get_view_control()
# Update the geometry and view
vis.update_geometry(pcd)
vis.poll_events()
vis.update_renderer()
print(1)
time.sleep(2)
# Change the view angle
ctr.rotate(100.0, 0.0)  # Rotate 30 degrees around the current view

# Update the geometry and view after changing the angle
vis.update_geometry(pcd)
vis.poll_events()
vis.update_renderer()
time.sleep(2)

# Close the window
vis.destroy_window()