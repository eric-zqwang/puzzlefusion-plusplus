from renderer.myrenderer import MyRenderer
import os
import hydra
import json
import bpy


def render_results(cfg, renderer: MyRenderer):
    save_dir = cfg.renderer.output_path
    
    sampled_files = renderer.sample_data_files()

    # sampled_files = ["1"]

    for file in sampled_files:
        transformation, gt_transformation, acc, init_pose = renderer.load_transformation_data(file)
        
        parts = renderer.load_mesh_parts(file, gt_transformation, init_pose)
        
        save_path = f"./BlenderToolBox_render/{save_dir}/{file}"
        os.makedirs(save_path, exist_ok=True)

        renderer.save_img(parts, gt_transformation, gt_transformation, init_pose, os.path.join(save_path, "gt.png"))

        frame = 0

        # bpy.ops.wm.save_mainfile(filepath=save_path + "test" + '.blend')

        for i in range(transformation.shape[0]):
            renderer.render_parts(
                parts, 
                gt_transformation, 
                transformation[i], 
                init_pose, 
                frame,
            )
            frame += 1


        imgs_path = os.path.join(save_path, "imgs")
        os.makedirs(imgs_path, exist_ok=True)
        renderer.save_video(imgs_path=imgs_path, video_path=os.path.join(save_path, "video.mp4"), frame=frame)
        renderer.clean()
        

@hydra.main(config_path="../config", config_name="auto_aggl")
def main(cfg):
    renderer = MyRenderer(cfg)
    render_results(cfg, renderer)



if __name__ == "__main__":
    main()