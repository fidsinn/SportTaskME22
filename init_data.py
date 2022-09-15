import os
import time
import os.path as osp
import gc

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
    # print('\nPerforming Human Detection for each frame')
    # prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= det_score_thr]
        results.append(result)
        # prog_bar.update()
    return results


def pose_inference(model_pos, frame_paths, det_results):
    model = model_pos
    ret = []
    # print('\nPerforming Human Pose Estimation for each frame')
    # prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        # prog_bar.update()
    return ret

# extract rgb skeleton and skeleton + rgb frames
def get_skeleton_frames(model_det, model_pos, frame_paths):
    # set detection parameters 
    det_score_thr = 0.9
    num_frame = len(frame_paths)
    # Get Human detection results
    det_results = detection_inference(model_det, det_score_thr, frame_paths)
    torch.cuda.empty_cache()
    
    # Get pose detection results
    pose_results = pose_inference(model_pos, frame_paths, det_results)
    torch.cuda.empty_cache()

    # visualise skeleton on black background with same dimensions as video
    # TODO: remove static dimension and make them dynamicly set to in video dims
    vis_frames_s = []
    for i in range(num_frame):
        #print('frame:', i)
        if len(pose_results[i]) >= 1:
            vis_frames_s.append(vis_pose_result(model_pos, 
                            np.zeros(shape=[1080, 1920, 3]),
                            [{'no_box': [],'keypoints': pose_results[i][0]['keypoints']}],
                            radius = 14,
                            thickness=10))
        else:
            vis_frames_s.append(np.zeros(shape=[1080, 1920, 3]))

    # visualise skeleton on rgb
    vis_frames_srgb = []
    for i in range(num_frame):
        if len(pose_results[i]) >= 1:
            vis_frames_srgb.append(vis_pose_result(model_pos, 
                            frame_paths[i],
                            [{'no_box': [],'keypoints': pose_results[i][0]['keypoints']}],
                            radius = 14,
                            thickness=10))
        else:
            vis_frames_srgb.append(cv2.imread(frame_paths[i]))

    return vis_frames_s, vis_frames_srgb

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

# save frame with correct size at given path
def save_frame(frame, frame_width, frames_path, idx):
    img = cv2.resize(frame, (frame_width, frame.shape[0]*frame_width//frame.shape[1]))
    cv2.imwrite(os.path.join(frames_path, '%08d.png' % idx), img)

# check if folders exists and create if needed
def create_folders_if_not_exist(folder_path_list):
    for folder_path in folder_path_list:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)



# create a working tree with skeleton, rgb and skeleton + rgb frames
# working_folder/
#   s/...
#   rgb/...
#   srgb/...
def create_working_tree(working_folder, source_folder, stream_design, frame_width=320, log=None):
    # Chrono
    start_time = time.time()
    batch_size = 500
    # Get all the videos and extract the RGB frames in the working_folder directory.
    
    list_of_videos = [f for f in getListOfFiles(os.path.join(source_folder)) if f.endswith('.mp4') and 's_' not in f and 'srgb_' not in f]
    # if stream_design == 'rgb':
    #    list_of_videos = [f for f in getListOfFiles(os.path.join(source_folder)) if ('s_' not in f and 'srgb_' not in f) or 'test' in f and f.endswith('.mp4') and '.' not in f]
    # elif stream_design == 's':
    #    list_of_videos = [f for f in getListOfFiles(os.path.join(source_folder)) if 's_' in f or 'test' in f and f.endswith('.mp4') and '.' not in f]
    # elif stream_design == 'srgb':
    #    list_of_videos = [f for f in getListOfFiles(os.path.join(source_folder)) if 'srgb_' in f or 'test' in f and f.endswith('.mp4') and '.' not in f]

    # init model for human detection
    # init once to save time
    model_det, model_pose = init_mmpose()
    
    for idx, video_path in enumerate(list_of_videos):
        if stream_design == 'rgb':
            pass
        elif stream_design == 's':
            video_path = video_path.replace('s_','')
        elif stream_design == 'srgb':
            video_path = video_path.replace('srgb_','')

        progress_bar(idx, len(list_of_videos), 'Frame + Pose extraction of:\n%s\n' % (video_path))

        frames_path_rgb = os.path.join(working_folder + '/rgb/', '/'.join(os.path.splitext(video_path)[0].split('/')[1:]))
        frames_path_s = os.path.join(working_folder + '/s/', '/'.join(os.path.splitext(video_path)[0].split('/')[1:]))
        frames_path_srgb = os.path.join(working_folder + '/srgb/', '/'.join(os.path.splitext(video_path)[0].split('/')[1:]))
        
        if stream_design == 'rgb':
            fp = frames_path_rgb
        elif stream_design == 's':
            fp = frames_path_s
        elif stream_design == 'srgb':
            fp = frames_path_srgb

        #if not os.path.exists(frames_path_rgb):
        if not os.path.exists(fp):
            
            create_folders_if_not_exist([frames_path_rgb, frames_path_s, frames_path_srgb])
            frame_extractor(video_path, 1080, frames_path_rgb)
            print('Frame extraction done in %ds\n' % (time.time() - start_time))
            
            # get paths to rgb frames to performe pose estimation
            paths_rgb = getListOfFiles(frames_path_rgb)
            paths_rgb.sort()
            num_frames = len(paths_rgb)

            # batch the frame extraction to be more frindly to memory 
            # and to avoid process termination by server
            num_of_batches = int((num_frames - (num_frames % batch_size)) / batch_size) 
            
            for batch_idx in range(0, num_of_batches):
                # extract the different frame types
                frames_s, frames_srgb = get_skeleton_frames(model_det, model_pose, paths_rgb[batch_idx * batch_size:(batch_idx + 1) * batch_size])          

                for i, frame in enumerate(frames_s):
                    save_frame(frame, frame_width, frames_path_s, (i + batch_idx * batch_size))
                
                for i, frame in enumerate(frames_srgb):
                    save_frame(frame, frame_width, frames_path_srgb, (i + batch_idx * batch_size))

            # go over rest of frames
            rest = num_frames % batch_size
            if rest > 0:
                frames_s, frames_srgb = get_skeleton_frames(model_det, model_pose, paths_rgb[num_frames - rest:num_frames])

                for i, frame in enumerate(frames_s):
                    save_frame(frame, frame_width, frames_path_s, (i + num_frames - rest))

                for i, frame in enumerate(frames_srgb):
                    save_frame(frame, frame_width, frames_path_srgb, (i + num_frames - rest))
                
        gc.collect()
        progress_bar(idx+1, len(list_of_videos), 'Frame + Pose extraction done in %ds\n' % (time.time() - start_time), 1, log=log)

    return 1