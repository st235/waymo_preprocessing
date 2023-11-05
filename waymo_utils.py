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
    """Processes a single waymo sequence.

    The code inspired by OpenPCDet:
    https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/waymo/waymo_utils.py.
    """
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
        frame_info['annos'] = annotations

        path_to_save_lidar = os.path.join(cur_save_dir, f"{frame_index:04d}.npy")
        num_points_of_each_lidar = __save_lidar_points(frame=frame,
                                                       path_to_save=path_to_save_lidar)
        frame_info['num_points_of_each_lidar'] = num_points_of_each_lidar

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
        'obj_ids': np.array(obj_ids),
        'name': np.array(obj_types),
        'difficulty': np.array(difficulties),
        'dimensions': np.array(dimensions),
        'location': np.array(locations),
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


def __convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose,
                                         ri_index=(0, 1)):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)

    for c in calibrations:
        points_single, cp_points_single, points_NLZ_single, points_intensity_single, points_elongation_single \
            = [], [], [], [], []
        for cur_ri_index in ri_index:
            range_image = range_images[c.name][cur_ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_NLZ = range_image_tensor[..., 3]
            range_image_intensity = range_image_tensor[..., 1]
            range_image_elongation = range_image_tensor[..., 2]
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.where(range_image_mask))
            points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
            points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
            points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
            cp = camera_projections[c.name][0]
            cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))

            points_single.append(points_tensor.numpy())
            cp_points_single.append(cp_points_tensor.numpy())
            points_NLZ_single.append(points_NLZ_tensor.numpy())
            points_intensity_single.append(points_intensity_tensor.numpy())
            points_elongation_single.append(points_elongation_tensor.numpy())

        points.append(np.concatenate(points_single, axis=0))
        cp_points.append(np.concatenate(cp_points_single, axis=0))
        points_NLZ.append(np.concatenate(points_NLZ_single, axis=0))
        points_intensity.append(np.concatenate(points_intensity_single, axis=0))
        points_elongation.append(np.concatenate(points_elongation_single, axis=0))

    return points, cp_points, points_NLZ, points_intensity, points_elongation


def __save_lidar_points(frame, path_to_save):
    ret_outputs = frame_utils.parse_range_image_and_camera_projection(frame)
    if len(ret_outputs) == 4:
        # The API changed in Waymo 1.6.0 and now returns 4 elements.
        range_images, camera_projections, seg_labels, range_image_top_pose = ret_outputs
    else:
        assert len(ret_outputs) == 3
        range_images, camera_projections, range_image_top_pose = ret_outputs

    points, cp_points, points_in_nlz_flag, points_intensity, points_elongation = __convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1)
    )

    points_all = np.concatenate(points, axis=0)
    points_in_nlz_flag = np.concatenate(points_in_nlz_flag, axis=0).reshape(-1, 1)
    points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
    points_elongation = np.concatenate(points_elongation, axis=0).reshape(-1, 1)

    # [n, 6] points.
    num_points_of_each_lidar = [point.shape[0] for point in points]
    save_points = np.concatenate([
        points_all, points_intensity, points_elongation, points_in_nlz_flag
    ], axis=-1).astype(np.float32)

    np.save(path_to_save, save_points)
    return num_points_of_each_lidar
