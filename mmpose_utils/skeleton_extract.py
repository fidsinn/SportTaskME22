# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil

import cv2
import mmcv
import numpy as np
import torch


from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result


def frame_extraction(video_path, short_side):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

        frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames


def detection_inference(det_config, det_checkpoint, device, det_score_thr, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(det_config, det_checkpoint, device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def pose_inference(pose_config, pose_checkpoint, device, frame_paths, det_results):
    model = init_pose_model(pose_config, pose_checkpoint, device)
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret


def get_skeleton_frames(video_path, short_side):
    directory_path = os.getcwd()
    # set detection parameters 
    config = mmcv.Config.fromfile(directory_path + '/skeleton_extract/configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint.py')
    config.merge_from_dict({})
    det_config =directory_path +  '/skeleton_extract/demo/faster_rcnn_r50_fpn_2x_coco.py'
    det_checkpoint = ('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth')
    device = 'cuda:0'
    det_score_thr = 0.9
    pose_config = directory_path + '/skeleton_extract/demo/hrnet_w32_coco_256x192.py'
    pose_checkpoint = ('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth')

    frame_paths, original_frames = frame_extraction(video_path, short_side)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # Get clip_len, frame_interval and calculate center index of each clip
    for component in config.data.test.pipeline:
        if component['type'] == 'PoseNormalize':
            component['mean'] = (w // 2, h // 2, .5)
            component['max_value'] = (w, h, 1.)
    
    # Get Human detection results
    det_results = detection_inference(det_config, det_checkpoint, device, det_score_thr, frame_paths)
    torch.cuda.empty_cache()
    
    pose_results = pose_inference(pose_config, pose_checkpoint, device, frame_paths, det_results)
    torch.cuda.empty_cache()

    pose_model = init_pose_model(pose_config, pose_checkpoint, device)


    vis_frames_s = []
    for i in range(num_frame):
        vis_frames_s.append(vis_pose_result(pose_model, 
                        np.zeros(shape=[h, w, 3]),
                        [{'no_box': [],'keypoints': pose_results[i][0]['keypoints']}],
                        radius = 7,
                        thickness=5))

    vis_frames_srgb = []
    for i in range(num_frame):
        vis_frames_srgb.append(vis_pose_result(pose_model, 
                        frame_paths[i],
                        [{'no_box': [],'keypoints': pose_results[i][0]['keypoints']}],
                        radius = 7,
                        thickness=5))
    
    frames_rgb = original_frames.copy()

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)

    return vis_frames_s, vis_frames_srgb, frames_rgb
