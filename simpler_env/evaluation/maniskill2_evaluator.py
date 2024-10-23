"""
Evaluate a model on ManiSkill2 environment.
"""

import os
import sys
import numpy as np
from transforms3d.euler import quat2euler

from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video

def stage1_subgoal_constraint1(end_effector, keypoints):
    """Align end-effector with the carrot's center."""
    carrot_center = keypoints[0]  # Assuming keypoint[0] is carrot center
    path_cost = np.linalg.norm(end_effector - carrot_center)
    return path_cost

def stage1_collision_constraint1(end_effector, keypoints):
    """Ensure the end-effector approaches from above."""
    carrot_center = keypoints[0]
    # Check if the z-coordinate of the end-effector is higher than the carrot's z-coordinate
    collision_cost = 0 if end_effector[2] > carrot_center[2] else 1  # Penalize if below the carrot
    return collision_cost

def stage1_grasp_constraint(grasp_status, is_src_obj_grasped):
    """Grasp the carrot."""
    grasp_cost = 0 if is_src_obj_grasped else 1  # Grasp cost is incurred when the carrot is grasped
    return grasp_cost


### Stage 2: Move carrot to plate
# The carrot must stay grasped and avoid collisions.
def stage2_grasp_constraint(grasp_status, is_src_obj_grasped, src_on_target):
    """Ensure the carrot remains grasped during the move."""
    grasp_cost = 0 if is_src_obj_grasped else 1  # Carrot must remain grasped
    return grasp_cost

def stage2_collision_constraint(end_effector, keypoints):
    """Ensure the carrot is aligned above the plate."""
    carrot_center = keypoints[0]
    plate_center = keypoints[1]  # Assuming keypoint[1] is the plate center
    collision_cost = np.linalg.norm(carrot_center[:2] - plate_center[:2])  # Only consider x and y axes
    return collision_cost


### Stage 3: Drop carrot on plate
# Ensure the carrot is placed on the plate and avoid collision.
def stage3_path_constraint(end_effector, keypoints):
    """Place the carrot on the plate."""
    carrot_center = keypoints[0]
    plate_center = keypoints[1]
    path_cost = np.linalg.norm(carrot_center - plate_center)  # Ensure carrot is on the plate center
    return path_cost

def stage3_collision_constraint(end_effector, keypoints):
    """Ensure end-effector moves away after placing the carrot."""
    carrot_center = keypoints[0]
    # Check if the end-effector moves above and away after placing
    collision_cost = 0 if end_effector[2] > carrot_center[2] else 1
    return collision_cost
    

def cal_cost(end_effector, keypoints, stage, info):

    cost={}
    grasp_status=info['is_src_obj_grasped']
    is_src_obj_grasped=info['is_src_obj_grasped']
    src_on_target=info['src_on_target']

    if(stage==1):
      cost['path_cost']=stage1_subgoal_constraint1(end_effector, keypoints)
      cost['col_cost']=stage1_collision_constraint1(end_effector, keypoints)
      cost['grasp_cost']=stage1_grasp_constraint(grasp_status, is_src_obj_grasped)


    elif(stage==2):
      cost['grasp_cost']=stage2_grasp_constraint(grasp_status, is_src_obj_grasped, src_on_target)
      cost['col_cost']=stage2_collision_constraint(end_effector, keypoints)


    elif(stage==3):
      cost['path_cost']=stage3_path_constraint(end_effector, keypoints)
      cost['col_cost']=stage3_collision_constraint(end_effector, keypoints)
    print(cost)
    cost_sum=0
    for k,v in cost.items():
      cost_sum+=v
    return cost_sum,cost





def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
):
    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, reset_info = env.reset(options=env_reset_options)
    print("obs:",obs['agent']['controller']['arm'])
    print("reset_source",reset_info['episode_source_obj_init_pose_wrt_robot_base'])
    print('reset_target',reset_info['episode_target_obj_init_pose_wrt_robot_base'])
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.unwrapped.is_final_subtask() 

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.unwrapped.get_language_instruction()
    print(task_description)

    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    # Initialize model
    model.reset(task_description)

    timestep = 0
    success = "failure"
    success_time=0
    stage=1
    cost=0
    cost_sum_dict={
      'col_cost': 0,
      'grasp_cost': 0,
      'path_cost': 0,
    }
    # Step the environment
    while not (predicted_terminated or truncated):
        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        raw_action, action = model.step(image, task_description)
        predicted_actions.append(raw_action)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated:
            if not is_final_subtask:
                # advance the environment to the next subtask
                predicted_terminated = False
                env.unwrapped.advance_to_next_subtask()

        # step the environment
        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
        )
        
        print("obs:",obs['agent']['controller']['arm'])
        print("reset_source",reset_info['episode_source_obj_init_pose_wrt_robot_base'])
        print('reset_target',reset_info['episode_target_obj_init_pose_wrt_robot_base'])
        obs_pose=obs['agent']['controller']['arm']['target_pose'][:3]
        source_pose=reset_info['episode_source_obj_init_pose_wrt_robot_base']
        tar_pose=reset_info['episode_target_obj_init_pose_wrt_robot_base']
        if(info["moved_correct_obj"]==False):
            stage=1    

        elif(info["moved_correct_obj"]==True and info['is_src_obj_grasped']==True):
            stage=2

        elif(info['is_src_obj_grasped']==True and info['consecutive_grasp']==True and info["src_on_target"]== False):
            stage=3
        keypoints=[np.array(source_pose.p),np.array(tar_pose.p)]
        print(obs_pose,keypoints)
        cost_step,cost_dict=cal_cost(end_effector=obs_pose,keypoints=keypoints,stage=stage,info=info)
        cost+=cost_step
        for k,v in cost_dict.items():
          cost_sum_dict[k]+=v
        success = "success" if done else "failure"
        if info['success']==True:
          success_time +=1
          print(success_time)
          #print(obs)
          print(timestep,info)
        new_task_description = env.unwrapped.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        is_final_subtask = env.unwrapped.is_final_subtask()

        print(timestep, info)
        if success_time>=3:
          break
        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images.append(image)
        timestep += 1

    episode_stats = info.get("episode_stats", {})
    print("cost:",cost)
    print(cost_sum_dict)
    # save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    cost_int=int(cost)
    for k,v in cost_sum_dict.items():
        video_name+=f'{k}_{v}_'
    video_name=video_name+f"cost_{cost_int}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)

    # save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success"


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []

    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        success_arr.append(run_maniskill2_eval_single_episode(obj_episode_id=obj_episode_id, **kwargs))
                else:
                    raise NotImplementedError()

    return success_arr
