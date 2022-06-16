import os
import time
import os.path as osp
import shutil

import cv2
import mmcv
import numpy as np
import torch


from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result

from utils import *


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


def detection_inference(model,  det_score_thr, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    print('Performing Human Detection for each frame\n')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def pose_inference(model_pos, frame_paths, det_results):
    model = model_pos
    ret = []
    print('Performing Human Pose Estimation for each frame\n')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret

# extract rgb skeleton and skeleton + rgb frames
def get_skeleton_frames(model_det, model_pos, video_path, short_side):
    # set detection parameters 
    det_score_thr = 0.9
    frame_paths, original_frames = frame_extraction(video_path, short_side)
    num_frame = len(frame_paths)
    
    # Get Human detection results
    det_results = detection_inference(model_det, det_score_thr, frame_paths)
    torch.cuda.empty_cache()
    
    # Get pose detection results
    pose_results = pose_inference(model_pos, frame_paths, det_results)
    torch.cuda.empty_cache()

    # visualise skeleton on black background
    vis_frames_s = []
    for i in range(num_frame):
        vis_frames_s.append(vis_pose_result(model_pos, 
                        np.zeros(shape=[h, w, 3]),
                        [{'no_box': [],'keypoints': pose_results[i][0]['keypoints']}],
                        radius = 7,
                        thickness=5))

    # visualise skeleton on rgb
    vis_frames_srgb = []
    for i in range(num_frame):
        vis_frames_srgb.append(vis_pose_result(model_pos, 
                        frame_paths[i],
                        [{'no_box': [],'keypoints': pose_results[i][0]['keypoints']}],
                        radius = 7,
                        thickness=5))
    
    # copy orignal frames so they wont get deleted when cleaning tmp files
    frames_rgb = original_frames.copy()

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)

    return vis_frames_s, vis_frames_srgb, frames_rgb

# init mmpose models with default values
def init_mmpose():
    # Attributes for mmpose initalisation
    det_config = 'mmpose_utils/demo/faster_rcnn_r50_fpn_2x_coco.py'
    det_checkpoint = ('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth')
    device = 'cuda:0'
    pose_config = 'mmpose_utils/demo/hrnet_w32_coco_256x192.py'
    pose_checkpoint = ('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth')


    model_det = init_detector(det_config, det_checkpoint, device)
    model_pos = init_pose_model(pose_config, pose_checkpoint, device)

    return model_det, model_pos

# creare a working tree with skeleton, rgb and skeleton + rgb frames
# working_folder/
#   s/...
#   rgb/...
#   srgb/...
def create_working_tree(working_folder, source_folder, frame_width=320, log=None):
    # Chrono
    start_time = time.time()
    # Get all the videos and extract the RGB frames in the working_folder directory.
    list_of_videos = [f for f in getListOfFiles(os.path.join(source_folder)) if f.endswith('.mp4') and 's_' not in f ]

    # init model for human detection
    # init once to save time
    model_det, model_pose = init_mmpose()

    for idx, video_path in enumerate(list_of_videos):
        progress_bar(idx, len(list_of_videos), 'Frame extraction of %s' % (video_path))
        frames_path_s = os.path.join(working_folder + '/s/', '/'.join(os.path.splitext(video_path)[0].split('/')[1:]))
        frames_path_srgb = os.path.join(working_folder + '/srgb/', '/'.join(os.path.splitext(video_path)[0].split('/')[1:]))
        frames_path_rgb = os.path.join(working_folder + '/rgb/', '/'.join(os.path.splitext(video_path)[0].split('/')[1:]))
        if not os.path.exists(frames_path_s) or  not os.path.exists(frames_path_srgb) or not os.path.exists(frames_path_rgb):
            # extract the different frame types
            frames_s, frames_srgb, frames_rgb  = get_skeleton_frames(model_det, model_pose, video_path, 1080)

            if not os.path.exists(frames_path_s):
                os.makedirs(frames_path_s)
                for i, frame in enumerate(frames_s):
                    s = cv2.resize(frame, (frame_width, frame.shape[0]*frame_width//frame.shape[1]))
                    cv2.imwrite(os.path.join(frames_path_s, '%08d.png' % i), s)

            if not os.path.exists(frames_path_srgb):
                os.makedirs(frames_path_srgb)
                for i, frame in enumerate(frames_srgb):
                    srgb = cv2.resize(frame, (frame_width, frame.shape[0]*frame_width//frame.shape[1]))
                    cv2.imwrite(os.path.join(frames_path_srgb, '%08d.png' % i), srgb)

            if not os.path.exists(frames_path_rgb):
                os.makedirs(frames_path_rgb)
                for i, frame in enumerate(frames_rgb):
                    rgb = cv2.resize(frame, (frame_width, frame.shape[0]*frame_width//frame.shape[1]))
                    cv2.imwrite(os.path.join(frames_path_rgb, '%08d.png' % i), rgb)

        progress_bar(idx+1, len(list_of_videos), 'Frame extraction done in %ds' % (time.time()-start_time), 1, log=log)


    return 1