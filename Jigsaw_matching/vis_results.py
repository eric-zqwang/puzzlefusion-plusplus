import numpy as np
import torch
import trimesh
import os
import re
from xgutils.vis import visutil
from xgutils.vis.fresnelvis import FresnelRenderer
import hydra
import glob
from pytorch3d import transforms


# Set 3: Analogous Colors
set3 = [
    (227, 38, 54),     # Crimson
    (255, 140, 0),     # Dark Orange
    (255, 165, 0),     # Orange
    (255, 195, 0),     # Orange Peel
    (255, 225, 0),     # Canary Yellow
    (255, 255, 0),     # Yellow
    (218, 165, 32),    # Golden Rod
    (188, 143, 143),   # Rosy Brown
    (205, 92, 92),     # Indian Red
    (188, 143, 143),   # Rosy Brown
    (244, 164, 96),    # Sandy Brown
    (210, 105, 30),    # Chocolate
    (139, 69, 19),     # Saddle Brown
    (165, 42, 42),     # Brown
    (128, 0, 0),       # Maroon
    (85, 107, 47),     # Dark Olive Green
    (107, 142, 35),    # Olive Drab
    (124, 252, 0),     # Lawn Green
    (127, 255, 0),     # Chartreuse
    (173, 255, 47),    # Green Yellow
]


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]


def load_transformation_data(data_dir, file):

    path = f"{data_dir}/{file}"
    # Pattern to match the file
    predict_pattern = "predict_*.npy"

    # Find the first file that matches the predict pattern
    predict_files = glob.glob(os.path.join(path, predict_pattern))

    predict_file_path = predict_files[0]

    acc = os.path.basename(predict_file_path).split('predict_')[-1].split('.npy')[0]

    # Load the predict.npy file
    transformation = np.load(predict_file_path)

    gt_transformation = np.load(f"{data_dir}/{file}/gt.npy")
    return transformation, gt_transformation, acc


def normalize_to_centroid(vertices, translation):
    centroid = np.mean(vertices, axis=0)
    return vertices - translation


def rotate_points(points, quaternion):
    points = torch.tensor(points, dtype=torch.float32)
    quaternion = torch.tensor(quaternion, dtype=torch.float32)
    # print(quaternion.shape)
    quat_inverse = transforms.quaternion_invert(quaternion)
    points = transforms.quaternion_apply(quat_inverse, points)
    return points.cpu().numpy()


def load_mesh_parts(file_path, transformation):
    mesh_dir_path = os.path.join("../Breaking-Bad-Dataset.github.io/data/", file_path)
    obj_files = [file for file in os.listdir(mesh_dir_path) if file.endswith('.obj')]
    parts = []
    obj_files.sort()

    count = 0
    for i, obj_file in enumerate(obj_files):
        full_path_to_obj = os.path.join(mesh_dir_path, obj_file)
        mesh = trimesh.load(full_path_to_obj)
        translation = transformation[count, :3]
        vert = normalize_to_centroid(np.array(mesh.vertices), translation)
        vert = rotate_points(vert, transformation[i, 3:])

        part = {"vert": vert, "face": np.array(mesh.faces)}
        parts.append(part)
        count += 1
    
    return parts


def render_parts(parts, camera_kwargs, trans, **kwargs):
    renderer = FresnelRenderer(camera_kwargs, **kwargs, lights="lightbox")

    rotation_matrix = trimesh.transformations.rotation_matrix(
        np.radians(-90), [1, 0, 0], point=[0, 0, 0]
    )

    for i,shape in enumerate(parts):
    
        translation = trans[i][:3]
        quat = trans[i][3:]

        # normalized to unit quaternion
        quat = quat / np.linalg.norm(quat)

        vert = shape["vert"]
    
        vert = transforms.quaternion_apply(torch.tensor(quat, dtype=torch.float32),
                                            torch.tensor(vert, dtype=torch.float32),).cpu().numpy()

        vert = vert + translation

        vert = vert.dot(rotation_matrix[:3, :3].T)

        # color = fresnelvis.unique_colors[i]
        color = np.array(set3[i]) / 255.0
        renderer.add_mesh(vert, face=shape["face"], color=color,
            solid=0., roughness=.2, specular=.8, spec_trans=0., metal=0.2
        )

    img = renderer.render(**kwargs)
    return img




def save_img(parts, camera_kwargs, gt_transformation, render_kwargs, file, save_dir):
    img = render_parts(parts, camera_kwargs, gt_transformation, **render_kwargs)
    visutil.saveImg(target=f"./render_results/{save_dir}/{file}.png", img=img)


import random
import pdb


# save_dir = "DDPM_100_val"


def sample_data_files(directory, num_samples=8000):
    # List all files in the given directory
    all_files = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    # Randomly sample 'num_samples' files
    sampled_files = random.sample(all_files, min(num_samples, len(all_files)))
    return sampled_files

import argparse
import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_dir", type=str, default="inference")
    parser.add_argument("--save_dir", type=str, default="render_results_by_id")
    args = parser.parse_args()

    os.makedirs(f"./render_results/{args.save_dir}", exist_ok=True)
    
    data_dir = args.inference_dir

    sampled_files = sample_data_files(data_dir)
    # increase order from 0 to max
    sampled_files.sort(key=natural_sort_key)

    for file in sampled_files:
        transformation, gt_transformation, acc = load_transformation_data(data_dir, file)
        with open(f"{data_dir}/{file}/mesh_file_path.txt", "r") as f:
            file_path = f.read().strip()

        if file == "11":
            print("here")
            # pdb.set_trace()

        parts = load_mesh_parts(file_path, gt_transformation)
        camera_kwargs = {
            'camera_type': 'orthographic',
            'fit_camera': False,
            'camPos': (1, 1, 1),
            'camLookat': (0, 0, 0),
            'camUp': (0, 1, 0),
            'camHeight': 2.2,
            'resolution': (256, 256),
            'samples': 32
        }

        render_kwargs = {"preview": True, "shadow_catcher": False}
        # pdb.set_trace()
        # save_img(parts, camera_kwargs, gt_transformation, render_kwargs, file, save_dir=args.save_dir)
        save_img(parts, camera_kwargs, transformation, render_kwargs, f"{file}", save_dir=args.save_dir)

        print(f"Saved images for {file}")
        



if __name__ == "__main__":
    main()
