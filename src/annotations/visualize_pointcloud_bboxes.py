import open3d as o3d


pcd_file = "data/CODa/static_map/CODa.pcd"

pcd = o3d.io.read_point_cloud(pcd_file)

o3d.visualization.draw_geometries([pcd], window_name="Static Map")
