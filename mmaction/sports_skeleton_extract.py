# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time

import cv2
import mmcv
import numpy as np
import torch
from mmcv import DictAction

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result

import moviepy.editor as mpy

import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Skeleton extraction')
    parser.add_argument('design', help='design (either s or srgb)')
    parser.add_argument('dir', help='input directory')
    #parser.add_argument('out_dir', help='output filename')
    parser.add_argument(
        '--config',
        default=('configs/skeleton/posec3d/'
                 'slowonly_r50_u48_240e_ntu120_xsub_keypoint.py'),
        help='skeleton model config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'slowonly_r50_u48_240e_ntu120_xsub_keypoint/'
                 'slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth'),
        help='skeleton model checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--label-map',
        default='tools/data/skeleton/label_map_ntu120.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


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


def detection_inference(args, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def pose_inference(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
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


def main():
    start_time = time.time()
    args = parse_args()

    # design = os.listdir(args.design)
    # if design = 's':
    #     stream_prefix = 's_'
    # elif design = 'srgb':
    #     stream_prefix = 'srgb_'
    design = args.design
    #videos = os.listdir(args.dir)
    videos = []
    videos_process_counter = 0
    if not (design == 's' or design == 'srgb'):
        sys.exit('\"Error: Enter design of s or srgb\"')
    for subdir, dirs, files in os.walk(args.dir):
        for file in files:
            if file.lower().endswith('.mp4'):
                videos.append(os.path.join(subdir,file))
    for subdir, dirs, files in os.walk(args.dir):
        for vid_file in files:
    #for vid_in in videos:
        #vid_in_path = args.dir + '/' + vid_in
            if file.lower().endswith('.mp4'):
                vid_in_path = os.path.join(subdir,vid_file)
                print('Processing: ' + vid_in_path)
                
                frame_paths, original_frames = frame_extraction(vid_in_path,
                                                                args.short_side)
                num_frame = len(frame_paths)
                h, w, _ = original_frames[0].shape

                # Get clip_len, frame_interval and calculate center index of each clip
                config = mmcv.Config.fromfile(args.config)
                config.merge_from_dict(args.cfg_options)
                for component in config.data.test.pipeline:
                    if component['type'] == 'PoseNormalize':
                        component['mean'] = (w // 2, h // 2, .5)
                        component['max_value'] = (w, h, 1.)

                # Load label_map
                label_map = [x.strip() for x in open(args.label_map).readlines()]

                # Get Human detection results
                det_results = detection_inference(args, frame_paths)
                torch.cuda.empty_cache()
            
                pose_results = pose_inference(args, frame_paths, det_results)
                torch.cuda.empty_cache()

                pose_model = init_pose_model(args.pose_config, 
                                            args.pose_checkpoint,
                                            args.device)
                
                if design == 's':
                    vis_frames = [
                        vis_pose_result(pose_model, 
                                        np.zeros(shape=[h, w, 3]),
                                        [{'bbox': [-1,-1,-1,-1,0],'keypoints': pose_results[i][0]['keypoints']}],
                                        radius = 7,
                                        thickness=5) #subtracts the box
                        for i in range(num_frame)
                    ]
                    print('vis_frames type:',type(vis_frames))
                elif design == 'srgb':
                    vis_frames = [
                        vis_pose_result(pose_model, 
                                        frame_paths[i],
                                        [{'bbox': [-1,-1,-1,-1,0],'keypoints': pose_results[i][0]['keypoints']}],
                                        radius = 7,
                                        thickness=5)
                        for i in range(num_frame)
                    ]
                    print('vis_frames type:',type(vis_frames))
                vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=100)
                #vid.write_videofile(args.out_dir + '/{}_'.format(design) + vid_in, remove_temp=True)
                #vid.write_videofile('{}/{}_{}'.format(subdir, design, file), remove_temp=True)
                vid.write_videofile(os.path.join(subdir,'{}_'.format(design)+file), remove_temp=True)

                tmp_frame_dir = osp.dirname(frame_paths[0])
                shutil.rmtree(tmp_frame_dir)
                print('Processed file {}'.format(os.path.join(subdir,'{}'.format(design)+file)))
                videos_process_counter += 1
                print('Processed {}/{} videos ({}%)'.format(videos_process_counter, len(videos), round((videos_process_counter/len(videos)),2)*100))
                print('Actual processing time: {}min'.format(round(((time.time()-start_time)/60),2)))

if __name__ == '__main__':
    main()