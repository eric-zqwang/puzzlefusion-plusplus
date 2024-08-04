import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import glob
import random
import blendertoolbox as bt
import bpy
import os
import pdb
from mathutils import Quaternion, Vector, Matrix
import subprocess


class MyRenderer:
    def __init__(self, cfg):
        super(MyRenderer, self).__init__()
        self.cfg = cfg
        self.mesh_path = cfg.renderer.mesh_path
        self.output_path = cfg.renderer.output_path
        self.inference_path = os.path.join(cfg.experiment_output_path, "inference", cfg.inference_dir)
        self.num_samples = cfg.renderer.num_samples
        self.category = cfg.renderer.category
        self.min_parts = cfg.renderer.min_parts
        self.max_parts = cfg.renderer.max_parts
        self.use_GPU = True
        self.scale = (1, 1, 1)
        self._init_param()
    
    
    def _init_param(self):
        # init blender
        bt.blenderInit(
            self.cfg.renderer.blender.imgRes_x,
            self.cfg.renderer.blender.imgRes_y,
            self.cfg.renderer.blender.numSamples,
            self.cfg.renderer.blender.exposure,
            self.cfg.renderer.blender.use_GPU
        )

        # set shadding
        bpy.ops.object.shade_smooth() # Option1: Gouraud shading
        # bpy.ops.object.shade_flat() # Option2: Flat shading
        # bt.edgeNormals(mesh, angle = 10) # Option3: Edge normal shading
        
        ## set invisible plane (shadow catcher)
        bt.invisibleGround(location=(0,0,-1), shadowBrightness=0.9)

        # set camera
        self.cam = bt.setCamera(
            camLocation=self.cfg.renderer.camera_kwargs.camPos,
            lookAtLocation=self.cfg.renderer.camera_kwargs.camLookat,
            focalLength=self.cfg.renderer.camera_kwargs.focalLength
        )

        self.location_offset = Vector(((-0.57, 0, 0.242)))

        # set light
        self.sun = bt.setLight_sun(
            rotation_euler=self.cfg.renderer.light.lightAngle,
            strength=self.cfg.renderer.light.strength,
            shadow_soft_size=self.cfg.renderer.light.shadowSoftness
        )
        bt.setLight_ambient(color=(0.2,0.2,0.2,1)) 
        bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')


    def assign_scale(self, scale):
        self.scale = scale
    
    def assign_invisible_plane_location(self, location):
        self.invisible_plane_location = location

    def sample_data_files(self):
        """
        sample data need to satisy the following conditions:
        1. self.min_part <= num_part <= self.max_part
        2. category == self.category
        """
        all_files = [f for f in os.listdir(self.inference_path) if os.path.isdir(os.path.join(self.inference_path, f))]
        
        data_list = []
        for i in range(len(all_files)):
            mesh_data_path = self._read_mesh_file(all_files[i])
            mesh_data_path = os.path.join(self.mesh_path, mesh_data_path)
            
            if self.category == "all" or self.category == "":
                num_parts = len(os.listdir(mesh_data_path))    
                if self.min_parts <= num_parts <= self.max_parts:
                    data_list.append(all_files[i])
                    
            if self.category.lower() in mesh_data_path.lower():
                num_parts = len(os.listdir(mesh_data_path))    
                if self.min_parts <= num_parts <= self.max_parts:
                    data_list.append(all_files[i])

        if self.num_samples != -1:
            random.seed(42)
            data_list = random.sample(data_list, min(self.num_samples, len(data_list)))
        return data_list
    

    def load_transformation_data(self, file):
        inference_data_path = f"{self.inference_path}/{file}"
        predict_pattern = "predict_*.npy"
        # Find the first file that matches the predict pattern
        predict_files = glob.glob(os.path.join(inference_data_path, predict_pattern))
        predict_file_path = predict_files[0]
        acc = os.path.basename(predict_file_path).split('predict_')[-1].split('.npy')[0]
        # Load the predict.npy file
        transformation = np.load(predict_file_path)
        gt_transformation = np.load(f"{inference_data_path}/gt.npy")
        init_pose = np.load(f"{inference_data_path}/init_pose.npy")
        return transformation, gt_transformation, acc, init_pose
    
    def load_edge_info_data(self, file, iter):
        inference_data_path = f"{self.inference_path}/{file}"
        # check if the file exists
        if not os.path.exists(f"{inference_data_path}/aggl_{iter}.npz"):
            return None
        edge_info = np.load(f"{inference_data_path}/aggl_{iter}.npz", allow_pickle=True)
        return edge_info

    def _read_mesh_file(self, file):
        with open(f"{self.inference_path}/{file}/mesh_file_path.txt", "r") as f:
            file_path = f.read()
        return file_path


    def load_mesh_parts(self, file, transformation, init_pose):
        file_path = self._read_mesh_file(file)

        mesh_dir_path = os.path.join(self.mesh_path, file_path)
        obj_files = [file for file in os.listdir(mesh_dir_path) if file.endswith('.obj')]

        parts = []
        obj_files.sort()

        for i, obj_file in enumerate(obj_files):
            meshPath = os.path.join(mesh_dir_path, obj_file)
            location = (-0.57, 0, 0.242)
            # location = (-0, 0, 0)
            rotation = (0, 0, 0)
            part = bt.readMesh(meshPath, location, rotation, scale=self.scale)
            # bt.subdivision(part, level = 2)
            RGB = np.array(self.cfg.renderer.colors[i]) / 255.0
            RGBA = np.append(RGB, 1.0)

            # meshColor = bt.colorObj(RGBA, 0.5, 1.0, 1.0, 0.0, 2.0)
            meshColor = bt.colorObj(RGBA)
            bt.setMat_plastic(part, meshColor)
            # AOStrength = 0.5
            # metalVal = 0.9
            # bt.setMat_metal(part, meshColor, AOStrength, metalVal)

            parts.append(part)
            part.rotation_mode = 'QUATERNION'
        
        return parts
    

    def render_parts(self, parts, gt_transformatoin, transformation, init_pose, frame=None):
        for i,shape in enumerate(parts):
            final_transformation = self.compute_final_transformation(init_pose, gt_transformatoin[i], transformation[i])
            shape.rotation_quaternion = final_transformation.to_quaternion()
            shape.location = self.location_offset + final_transformation.to_translation()
            
            if frame is not None:
                # print("Rotation Quaternion:", final_transformation.to_quaternion())  # Debug print
                # print("Location:", final_transformation.to_translation())  # Debug print


                shape.keyframe_insert(data_path="location", frame=frame)
                shape.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    
    def _get_init_locations(self, init_pose, gt_transformation):
        trans1 = Vector(-init_pose[:3])
        trans_mat1 = Matrix.Translation(trans1)
        rot_mat1 = Quaternion(init_pose[3:]).inverted().to_matrix().to_4x4()

        trans2 = Vector(-gt_transformation[:3])
        trans_mat2 = Matrix.Translation(trans2)
        rot_mat2 = Quaternion(gt_transformation[3:]).inverted().to_matrix().to_4x4()

        final_transformation = rot_mat2 @ trans_mat2  @ trans_mat1 @ rot_mat1
        return final_transformation
    

    def save_per_part(self, part, gt_transformation, init_pose, save_path):
        visibility = {obj: obj.hide_render for obj in bpy.context.scene.objects}
        for obj in bpy.context.scene.objects:
            if obj.name.startswith('piece'):
                obj.hide_render = obj != part

        final_transformation = self._get_init_locations(init_pose, gt_transformation)
        
        part.rotation_quaternion = final_transformation.to_quaternion()
        part.location = self.location_offset + final_transformation.to_translation()

        bt.renderImage(f"{save_path}", self.cam)

        for obj, vis in visibility.items():
            obj.hide_render = vis
        
    def find_cluster(self, clusters, index):
        return next((cluster for cluster in clusters if index in cluster), None)

    def render_connected_edges(self, parts, edge_info_data, save_path):
        
        classified_edges = edge_info_data["classified_edges"]
        cluster = edge_info_data['current_clusters']
        scores = edge_info_data['scores']

        for i, edges in enumerate(classified_edges):
            idx_1 = edges[0]
            idx_2 = edges[1]

            idx_1_cluster = self.find_cluster(cluster, idx_1)
            idx_2_cluster = self.find_cluster(cluster, idx_2)

            cluster_part = []
            for i in range(len(parts)):
                if i in idx_1_cluster:
                    cluster_part.append(parts[i])
                if i in idx_2_cluster:
                    cluster_part.append(parts[i])

            for obj in bpy.context.scene.objects:
                if obj.name.startswith('piece'):
                    hide_parts = obj not in cluster_part # if the part is not in the cluster, hide it
                    obj.hide_render = hide_parts

            bt.renderImage(f"{save_path}/{idx_1_cluster}_{idx_2_cluster}", self.cam)
            
            for obj in bpy.context.scene.objects:
                obj.hide_render = False
        


    def compute_final_transformation(self, init_pose, gt_transformation, transformation):

        trans1 = Vector(-init_pose[:3])
        trans_mat1 = Matrix.Translation(trans1)
        rot_mat1 = Quaternion(init_pose[3:]).inverted().to_matrix().to_4x4()

        trans2 = Vector(-gt_transformation[:3])
        trans_mat2 = Matrix.Translation(trans2)
        rot_mat2 = Quaternion(gt_transformation[3:]).inverted().to_matrix().to_4x4()

        trans3 = Vector(transformation[:3])
        trans_mat3 = Matrix.Translation(trans3)
        rot_mat3 = Quaternion(transformation[3:]).normalized().to_matrix().to_4x4()

        trans4 = Vector(init_pose[:3])
        trans_mat4 = Matrix.Translation(trans4)
        rot_mat4 = Quaternion(init_pose[3:]).to_matrix().to_4x4()

        # rotate -> translate -> translate -> rotate -> rotate -> translate
        final_transformation = rot_mat4 @ trans_mat4 @ trans_mat3 @ rot_mat3 @ rot_mat2 @ trans_mat2  @ trans_mat1 @ rot_mat1
        return final_transformation
        


    def save_video(self, imgs_path, video_path, frame):
        bt.renderAnimation(f"{imgs_path}/", self.cam, duration=frame)
        
        # Compile frames into a video using FFmpeg
        command = [
            'ffmpeg', 
            '-framerate', f'{frame / 8}',  
            '-i', f'{imgs_path}/%04d.png',  # Adjust the pattern based on how your frames are named
            '-vf', 'tpad=stop_mode=clone:stop_duration=2',  # Hold the last frame for 2 seconds
            '-c:v', 'libx264', 
            '-pix_fmt', 'yuv420p', 
            '-crf', '17',  # Adjust CRF (lower means higher quality)
            video_path
        ]
        subprocess.run(command, check=True)
        
        # Optional: Cleanup temporary frames
        # for file_name in os.listdir(temp_dir):
        #     os.remove(os.path.join(temp_dir, file_name))

        print(f"Video saved to {video_path}")


    def save_img(self, parts, gt_transformatoin, transformation, init_pose, save_path):
        self.render_parts(parts, gt_transformatoin, transformation, init_pose)
        # bt.renderImage(f"./render_results/{self.output_path}/{file}.png", self.cam)
        bt.renderImage(f"{save_path}", self.cam)
        

    def clean(self):
        for obj in bpy.data.objects:
            # Check if the object is a mesh
            if obj.type == 'MESH':
                # Select the mesh object
                obj.select_set(True)
            else:
                # Deselect other objects
                obj.select_set(False)
        bpy.ops.object.delete()

