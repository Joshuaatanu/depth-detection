import requests
import numpy as np
import open3d as o3d
from PIL import Image
from io import BytesIO


def estimate_depth_api(image_path):
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        response = requests.post(
            "https://api.roboflow.com/depth-estimation", files=files)

    if response.status_code == 200:
        depth_map = np.array(Image.open(
            BytesIO(response.content)).convert("L"))
        return depth_map
    else:
        print("Error:", response.text)
        return None


depth_map = estimate_depth_api("room.jpg")
if depth_map is not None:
    point_cloud = depth_to_point_cloud(depth_map)
    o3d.visualization.draw_geometries([point_cloud])
