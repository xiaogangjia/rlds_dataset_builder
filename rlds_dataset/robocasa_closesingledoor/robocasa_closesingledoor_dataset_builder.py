import json
import os

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import h5py
import einops
from pathlib import Path
import pickle

# horeka path: /hkfs/work/workspace/scratch/ll6323-david_dataset_2/robocasa_datasets/v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02/processed_demo_128_128.hdf5
raw_data_path = os.path.join("/hkfs/work/workspace/scratch/ll6323-david_dataset_2/robocasa_datasets/v0.1/single_stage", "kitchen_doors/CloseSingleDoor/2024-04-24/processed_demo_128_128.hdf5") #"/mnt/d/kit/masterarbeit/processed_demo_128_128.hdf5"

env_data = h5py.File(raw_data_path, "r")
env_data = env_data["data"]



class robocasa_turnonstove(tfds.core.GeneratorBasedBuilder):
    """
    Convert a Hugging Face dataset into a TFDS-style episodic dataset with metadata.
    """
    VERSION = tfds.core.Version('1.0.0')

    def __init__(self, **kwargs):
        """
        Args:
            dataset_name (str): Name of the Hugging Face dataset.
            episodes (List[int]): List of episode indices to load. Default is None (load all).
        """
        dataset_name = "robocasa_turnonstove"
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path

        super().__init__()

    def _info(self):
        """Define dataset information and features."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=f"Converted from the Hugging Face dataset {self.dataset_name}.",
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset(
                    {
                        'is_first': tf.bool,
                        'is_last': tf.bool,
                        'observation': tfds.features.FeaturesDict({
                            'state': tfds.features.Tensor(shape=(25,), dtype=tf.float32),
                            'robot0_agentview_left_image': tfds.features.Tensor(shape=(128, 128, 3), dtype=tf.uint8),
                            'robot0_agentview_right_image': tfds.features.Tensor(shape=(128, 128, 3), dtype=tf.uint8),
                            'robot0_eye_in_hand_image': tfds.features.Tensor(shape=(128, 128, 3), dtype=tf.uint8),
                            'robot0_agentview_left_point_cloud': tfds.features.Tensor(shape=(128, 128, 3), dtype=tf.float32),
                            'robot0_agentview_right_point_cloud': tfds.features.Tensor(shape=(128, 128, 3), dtype=tf.float32),
                            'robot0_eye_in_hand_point_cloud': tfds.features.Tensor(shape=(128, 128, 3), dtype=tf.float32),
                            'robot0_agentview_left_depth': tfds.features.Tensor(shape=(128, 128, 1), dtype=tf.float32),
                            'robot0_agentview_right_depth': tfds.features.Tensor(shape=(128, 128, 1), dtype=tf.float32),
                            'robot0_eye_in_hand_depth': tfds.features.Tensor(shape=(128, 128, 1), dtype=tf.float32),
                            'sampled_point_cloud': tfds.features.Tensor(shape=(1024, 6), dtype=tf.float32),
                            'uniform_sampled_point_cloud': tfds.features.Tensor(shape=(1024, 6), dtype=tf.float32),
                        }),
                        'action': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                        'reward': tfds.features.Tensor(shape=(), dtype=tf.float32),
                        'timestamp': tfds.features.Tensor(shape=(), dtype=tf.float32),
                        'frame_index': tfds.features.Tensor(shape=(), dtype=tf.int32),
                        'is_terminal': tfds.features.Tensor(shape=(), dtype=tf.bool),
                        'language_instruction': tfds.features.Text(),
                        'discount': tfds.features.Tensor(shape=(), dtype=tf.float32),
                        'metadata': tfds.features.FeaturesDict({
                            'episode_index': tfds.features.Tensor(shape=(), dtype=tf.int32)
                        }),
                    }
                ),
                'episode_metadata': tfds.features.FeaturesDict({
                    'episode_id': tfds.features.Tensor(shape=(), dtype=tf.int32),
                })
            }),
        )

    def _split_generators(self, dl_manager):
        """Specify dataset splits."""
        return {
            'train': self._generate_examples()
        }

    def _generate_examples(self):
        """Yield examples grouped by episodes."""
        path_list = get_all_file_names(self.raw_data_path)
        for path in path_list:
            data = process_episode_data(path)
            yield path, data



def process_episode_data(file_path):
    """
    Reads a .parquet file and converts it into a NumPy array.

    Parameters:
        file_path (str): The path to the .parquet file.

    Returns:
        numpy.ndarray: A NumPy array containing the data from the .parquet file.
    """
    try:
        # Read the parquet file into a pandas DataFrame
        # episode = pickle.load(open(file_path, 'rb'))
        # episode_data = episode['data']

        demo_length = env_data[file_path].attrs["num_samples"]

        lang = json.loads(env_data[file_path].attrs["ep_meta"])["lang"]

        actions = env_data[file_path]['actions']

        full_point_clouds = env_data[file_path]["obs"]["point_cloud"]

        # c=6, 0-2 stand for xyz, 3-5 stand for rgb
        rgbs = einops.rearrange(
            full_point_clouds[:, :, :],
            "t (num_cam h w) c -> t num_cam h w c",
            num_cam=3,
            h=128,
            w=128,
        )

        robot0_agentview_left_images = rgbs[:, 0, :, :, 3:]
        robot0_agentview_right_images = rgbs[:, 1, :, :, 3:]
        robot0_eye_in_hand_images = rgbs[:, 2, :, :, 3:]

        robot0_agentview_left_point_clouds = rgbs[:, 0, :, :, :3]
        robot0_agentview_right_point_clouds = rgbs[:, 1, :, :, :3]
        robot0_eye_in_hand_point_clouds = rgbs[:, 2, :, :, :3]

        robot0_agentview_left_depths = env_data[file_path]["obs"]['robot0_agentview_left_depth']
        robot0_agentview_right_depths = env_data[file_path]["obs"]["robot0_agentview_right_depth"]
        robot0_eye_in_hand_depths = env_data[file_path]["obs"]["robot0_eye_in_hand_depth"]

        sampled_point_clouds = env_data[file_path]["obs"]['sampled_point_cloud']
        uniform_sampled_point_clouds = env_data[file_path]["obs"]['uniform_sampled_point_cloud']

        robot0_eef_pos = env_data[file_path]["obs"]["robot0_eef_pos"]
        robot0_eef_quat = env_data[file_path]["obs"]["robot0_eef_quat"]

        robot0_gripper_qpos = env_data[file_path]["obs"]["robot0_gripper_qpos"]
        robot0_gripper_qvel = env_data[file_path]["obs"]["robot0_gripper_qvel"]

        robot0_joint_pos = env_data[file_path]["obs"]["robot0_joint_pos"]
        robot0_joint_vel = env_data[file_path]["obs"]["robot0_joint_vel"]

        steps = []

        for idx in range(demo_length):

            action = actions[idx][:7].astype(np.float32)

            robot0_agentview_left_image = robot0_agentview_left_images[idx].astype(np.uint8)
            robot0_agentview_right_image = robot0_agentview_right_images[idx].astype(np.uint8)
            robot0_eye_in_hand_image = robot0_eye_in_hand_images[idx].astype(np.uint8)

            robot0_agentview_left_point_cloud = robot0_agentview_left_point_clouds[idx].astype(np.float32)
            robot0_agentview_right_point_cloud = robot0_agentview_right_point_clouds[idx].astype(np.float32)
            robot0_eye_in_hand_point_cloud = robot0_eye_in_hand_point_clouds[idx].astype(np.float32)

            robot0_agentview_left_depth = robot0_agentview_left_depths[idx]
            robot0_agentview_right_depth = robot0_agentview_right_depths[idx]
            robot0_eye_in_hand_depth = robot0_eye_in_hand_depths[idx]

            sampled_point_cloud = sampled_point_clouds[idx].astype(np.float32)
            uniform_sampled_point_cloud = uniform_sampled_point_clouds[idx].astype(np.float32)

            eef_pos = robot0_eef_pos[idx]
            eef_quat = robot0_eef_quat[idx]
            gripper_qpos = robot0_gripper_qpos[idx]
            gripper_qvel = robot0_gripper_qvel[idx]
            joint_pos = robot0_joint_pos[idx]
            joint_vel = robot0_joint_vel[idx]

            # 7,7,2,2,3,4
            proprioceptive_state = np.concatenate((joint_pos, joint_vel, gripper_qpos, gripper_qvel, eef_pos, eef_quat), axis=0).astype(np.float32)

            steps.append({
                'is_first': idx == 0,
                'is_last': idx == demo_length - 1,
                'observation': {
                    'state': proprioceptive_state,
                    'robot0_agentview_left_image': robot0_agentview_left_image,
                    'robot0_agentview_right_image': robot0_agentview_right_image,
                    'robot0_eye_in_hand_image': robot0_eye_in_hand_image,
                    'robot0_agentview_left_point_cloud': robot0_agentview_left_point_cloud,
                    'robot0_agentview_right_point_cloud': robot0_agentview_right_point_cloud,
                    'robot0_eye_in_hand_point_cloud': robot0_eye_in_hand_point_cloud,
                    'robot0_agentview_left_depth': robot0_agentview_left_depth,
                    'robot0_agentview_right_depth': robot0_agentview_right_depth,
                    'robot0_eye_in_hand_depth': robot0_eye_in_hand_depth,
                    'sampled_point_cloud': sampled_point_cloud,
                    'uniform_sampled_point_cloud': uniform_sampled_point_cloud,
                },
                'action': action,
                'reward': 0.0,
                'language_instruction': lang,
                'is_terminal': idx == demo_length - 1,
                'discount': 1.0,
                'timestamp': idx,
                'frame_index': idx,
                'metadata': {'episode_index': 0}
            })
        return {'steps': steps, 'episode_metadata': {'episode_id': 0}}

    except Exception as e:
        print(f"An error occurred while reading the parquet file: {e}")
        return None


def get_all_file_names(path):
    """
    Get all file names under a specified path.

    Args:
        path (str): The directory path to search.

    Returns:
        List[str]: A list of file names under the specified path.
    """

        # Path to the folder you want to search in
    # folder_path = Path(path).resolve()  # resolve() makes the path absolute

    # Get all .pkl files in the specified folder
    # pkl_files = [str(file.absolute()) for file in folder_path.glob('*.pkl')]
    # demo_data = [env_data[demo] for demo in env_data]

    demo_keys = list(env_data.keys())

    return demo_keys

    # return [path]
    # return pkl_files


if __name__ == "__main__":
    folder_path = os.path.join("/hkfs/work/workspace/scratch/ll6323-david_dataset_2/robocasa_datasets/v0.1/single_stage", "kitchen_doors/CloseSingleDoor/2024-04-24/processed_demo_128_128.hdf5") #"/mnt/d/kit/masterarbeit/processed_demo_128_128.hdf5"

    pkl_files = get_all_file_names(folder_path)

    # print(pkl_files)

    for pkl_file in pkl_files:

        episode_data = process_episode_data(pkl_file)
        # print(pkl_file)
    # Load the dataset
    # ds = tfds.load("real_franka_fold")

    # for episode in ds['train']:
    #     for step in episode['steps']:
    #         print(step)
    #     break

