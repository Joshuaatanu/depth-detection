from transformers import DPTForDepthEstimation, DPTImageProcessor
from PIL import Image
import torch
import numpy as np
import open3d as o3d

MODEL_PATH = "models/dpt-large"

# Load Model
processor = DPTImageProcessor.from_pretrained(MODEL_PATH)
model = DPTForDepthEstimation.from_pretrained(MODEL_PATH).eval()


def estimate_depth(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        depth_map = outputs.predicted_depth.squeeze().cpu().numpy()

    return depth_map


def depth_to_point_cloud(depth_map):
    height, width = depth_map.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_map.flatten()
    valid = z > 0
    x = (u.flatten() - width // 2) * z / 500
    y = (v.flatten() - height // 2) * z / 500
    points = np.stack((x, y, z), axis=1)[valid]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud


# Example Usage
depth_map = estimate_depth("room.jpg")
point_cloud = depth_to_point_cloud(depth_map)
o3d.visualization.draw_geometries([point_cloud])
