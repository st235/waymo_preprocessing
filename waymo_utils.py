import os
import pickle
import numpy as np
import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils

try:
    tf.enable_eager_execution()
except:
    pass

WAYMO_CLASSES = ['Unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']


def process_single_sequence(sequence_file: str,
                            save_dir: str):
    sequences_meta = []
    sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]

    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
    cur_save_dir = os.path.join(save_dir, sequence_name)

    if not os.path.exists(cur_save_dir):
        os.makedirs(cur_save_dir)

    for frame_index, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        frame_info = dict()

        frame_info['frame_id'] = f"{sequence_name}_{frame_index:03d}"
        frame_info['metadata'] = {
            'context_name': frame.context.name,
            'timestamp_micros': frame.timestamp_micros
        }

        pc_info = {
            'num_features': 5,
            'lidar_sequence': sequence_name,
            'frame_index': frame_index
        }

        frame_info['point_cloud'] = pc_info

        image_info = {}
        for j in range(5):
            width = frame.context.camera_calibrations[j].width
            height = frame.context.camera_calibrations[j].height
            image_info.update({'image_shape_%d' % j: (height, width)})
        frame_info['image'] = image_info

        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        frame_info['pose'] = pose

        annotations = __generate_labels(frame=frame, pose=pose)
        frame_info['annotations'] = annotations

        path_to_save_lidar = os.path.join(cur_save_dir, f"{frame_index:04d}.npy")
        num_points_saved = __save_lidar_points(frame=frame,
                                               path_to_save=path_to_save_lidar)
        frame_info['num_points_saved'] = num_points_saved

        sequences_meta.append(frame_info)

    pkl_file = os.path.join(save_dir, f"{sequence_name}.pkl")
    with open(pkl_file, 'wb') as f:
        pickle.dump(sequences_meta, f)

    return sequences_meta


def __generate_labels(frame, pose):
    obj_types, obj_ids = [], []
    difficulties, dimensions, locations, heading_angles = [], [], [], []
    tracking_difficulty, speeds, accelerations, obj_ids = [], [], [], []
    num_points_in_gt = []

    for label in frame.laser_labels:
        class_type = label.type
        obj_ids.append(label.id)
        obj_types.append(WAYMO_CLASSES[class_type])

        box = label.box
        location = [box.center_x, box.center_y, box.center_z]
        locations.append(location)
        # LWH similar to OpenPCD.
        dimensions.append([box.length, box.width, box.height])
        heading_angles.append(box.heading)

        speeds.append([label.metadata.speed_x, label.metadata.speed_y])
        accelerations.append([label.metadata.accel_x, label.metadata.accel_y])

        difficulties.append(label.detection_difficulty_level)
        tracking_difficulty.append(label.tracking_difficulty_level)
        num_points_in_gt.append(label.num_lidar_points_in_box)

    annotations = {
        'ids': np.array(obj_ids),
        'types': np.array(obj_types),
        'difficulties': np.array(difficulties),
        'dimensions': np.array(dimensions),
        'locations': np.array(locations),
        'heading_angles': np.array(heading_angles),
        'tracking_difficulty': np.array(tracking_difficulty),
        'num_points_in_gt': np.array(num_points_in_gt),
        'speed_global': np.array(speeds),
        'accel_global': np.array(accelerations),
    }

    if len(annotations['ids']) > 0:
        global_speed = np.pad(annotations['speed_global'], ((0, 0), (0, 1)), mode='constant',
                              constant_values=0)  # (N, 3)
        speed = np.dot(global_speed, np.linalg.inv(pose[:3, :3].T))
        speed = speed[:, :2]

        gt_boxes_lidar = np.concatenate([
            annotations['locations'], annotations['dimensions'], annotations['heading_angles'][..., np.newaxis], speed],
            axis=1)
    else:
        gt_boxes_lidar = np.zeros((0, 9))

    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    return annotations


def __save_lidar_points(frame, path_to_save):
    ret_outputs = frame_utils.parse_range_image_and_camera_projection(frame)
    if len(ret_outputs) == 4:
        # The API changed in Waymo 1.6.0 and now returns 4 elements.
        range_images, camera_projections, seg_labels, range_image_top_pose = ret_outputs
    else:
        assert len(ret_outputs) == 3
        range_images, camera_projections, range_image_top_pose = ret_outputs

    point_cloud_0, _ = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections,
                                                                      range_image_top_pose, ri_index=0)

    point_cloud_1, _ = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections,
                                                                      range_image_top_pose, ri_index=1)

    points_all_0 = np.concatenate(point_cloud_0, axis=0)
    points_all_1 = np.concatenate(point_cloud_1, axis=0)

    # [n, 3] points.
    points_all = np.concatenate((points_all_0, points_all_1))
    np.save(path_to_save, points_all)

    return points_all.shape[0]
