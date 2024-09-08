import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
from typing import Any, Dict, List, Optional, Tuple

import dask.dataframe as dd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.utils.lidar_utils import (
    convert_lidar_pose_range_image_to_transformation,
    convert_range_image_to_cartesian)


def preprocess_sequence(sequence_idx, sequence_file, split):
    points_df = dd.read_parquet(sequence_file)
    labels_df = dd.read_parquet(f"/datasets_master/waymo_open_v_2_0_0/{split}/lidar_segmentation/{sequence_file.split('/')[-1][:-8]}.parquet")
    calibration_df = dd.read_parquet(f"/datasets_master/waymo_open_v_2_0_0/{split}/lidar_calibration/{sequence_file.split('/')[-1][:-8]}.parquet")
    pose_df = dd.read_parquet(f"/datasets_master/waymo_open_v_2_0_0/{split}/lidar_pose/{sequence_file.split('/')[-1][:-8]}.parquet")
    df = v2.merge(points_df, labels_df)
    df = v2.merge(df, calibration_df)
    df = v2.merge(df, pose_df)

    for frame, (_, row) in enumerate(iter(df.iterrows())):
        range_image_r1 = v2.LiDARComponent.from_dict(row).range_image_return1
        range_image_r2 = v2.LiDARComponent.from_dict(row).range_image_return2
        labels_r1 = v2.LiDARSegmentationLabelComponent.from_dict(row).range_image_return1
        labels_r2 = v2.LiDARSegmentationLabelComponent.from_dict(row).range_image_return2
        calibration = v2.LiDARCalibrationComponent.from_dict(row)
        points_r1 = convert_range_image_to_cartesian(
            range_image_r1,
            calibration,
            keep_polar_features=True
        )
        mask_r1 = range_image_r1.tensor[..., 0] > 0
        points_r1 = tf.gather_nd(
            points_r1,
            tf.compat.v1.where(mask_r1)
        ).numpy()[:, [3,4,5,1]]
        points_r2 = convert_range_image_to_cartesian(
            range_image_r2,
            calibration,
            keep_polar_features=True
        )
        mask_r2 = range_image_r2.tensor[..., 0] > 0
        points_r2 = tf.gather_nd(
            points_r2,
            tf.compat.v1.where(mask_r2)
        ).numpy()[:, [3,4,5,1]]
        labels_r1 = labels_r1.tensor[mask_r1].numpy()
        labels_r2 = labels_r2.tensor[mask_r2].numpy()
        # extrinsic = calibration.extrinsic.transform
        pose = v2.LiDARPoseComponent.from_dict(row).range_image_return1
        pose = convert_lidar_pose_range_image_to_transformation(pose).numpy().astype(np.float64)
        pose_global = np.mean(pose[mask_r1], 0)  # Here we assume that the pose is constant for the whole frame (not accounting for the rotation of the lidar)

        # p_W = np.matmul(pose[mask_r1], np.pad(points_r1, ((0, 0), (0, 1)), constant_values=1).reshape(-1,4,1))[...,0]
        # p_gW = np.matmul(pose_global, np.pad(points_r1, ((0, 0), (0, 1)), constant_values=1).reshape(-1,4,1))[...,0]
        # np.savetxt(f"/root/workspace/test_W_{_}.txt", p_W)
        # np.savetxt(f"/root/workspace/test_gW_{_}.txt", p_gW)
        # np.savetxt(f"/root/workspace/test_{_}.txt", points_r1)

        np.save(f"/datasets_local/waymo_v2_preprocessed/{split}/{sequence_idx}/lidar/frame_{frame}_r1.npy", np.ascontiguousarray(points_r1))
        np.save(f"/datasets_local/waymo_v2_preprocessed/{split}/{sequence_idx}/lidar/frame_{frame}_r2.npy", np.ascontiguousarray(points_r2))
        np.save(f"/datasets_local/waymo_v2_preprocessed/{split}/{sequence_idx}/labels/frame_{frame}_r1.npy", np.ascontiguousarray(labels_r1[:, 1]))
        np.save(f"/datasets_local/waymo_v2_preprocessed/{split}/{sequence_idx}/labels/frame_{frame}_r2.npy", np.ascontiguousarray(labels_r2[:, 1]))
        np.save(f"/datasets_local/waymo_v2_preprocessed/{split}/{sequence_idx}/instances/frame_{frame}_r1.npy", np.ascontiguousarray(labels_r1[:, 0]))
        np.save(f"/datasets_local/waymo_v2_preprocessed/{split}/{sequence_idx}/instances/frame_{frame}_r2.npy", np.ascontiguousarray(labels_r2[:, 0]))
        np.save(f"/datasets_local/waymo_v2_preprocessed/{split}/{sequence_idx}/calib/frame_{frame}.npy", pose_global)


if __name__ == "__main__":
    for split in ["training", "validation"]:
        list_files = glob.glob(f"/datasets_master/waymo_open_v_2_0_0/{split}/lidar/*.parquet")
        list_files = sorted(list_files)
        print(list_files)
        for i in range(len(list_files)):
            os.makedirs(f"/datasets_local/waymo_v2_preprocessed/{split}/{i}/lidar", exist_ok=True)
            os.makedirs(f"/datasets_local/waymo_v2_preprocessed/{split}/{i}/labels", exist_ok=True)
            os.makedirs(f"/datasets_local/waymo_v2_preprocessed/{split}/{i}/instances", exist_ok=True)
            os.makedirs(f"/datasets_local/waymo_v2_preprocessed/{split}/{i}/calib", exist_ok=True)
        
        for i, file in tqdm(enumerate(list_files), total=len(list_files)):
            preprocess_sequence(i, file, split)
