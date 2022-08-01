'''Verification script by Pierre-Etienne MARTIN dedicated to the Sport Task for MediaEval 2021'''
import os
import numpy as np
import argparse
import cv2, pdb
from xml.etree import ElementTree
import operator

dict_of_moves = ['Serve Forehand Backspin',
                'Serve Forehand Loop',
                'Serve Forehand Sidespin',
                'Serve Forehand Topspin',

                'Serve Backhand Backspin',
                'Serve Backhand Loop',
                'Serve Backhand Sidespin',
                'Serve Backhand Topspin',

                'Offensive Forehand Hit',
                'Offensive Forehand Loop',
                'Offensive Forehand Flip',

                'Offensive Backhand Hit',
                'Offensive Backhand Loop',
                'Offensive Backhand Flip',

                'Defensive Forehand Push',
                'Defensive Forehand Block',
                'Defensive Forehand Backspin',

                'Defensive Backhand Push',
                'Defensive Backhand Block',
                'Defensive Backhand Backspin']

'''
    For the given path, get the List of all files in the directory tree 
'''
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

def compute_iou(gt, prediction):
    intersection = np.logical_and(gt, prediction)
    union = np.logical_or(gt, prediction)
    iou_score = 1.*np.sum(intersection) / np.sum(union)
    return iou_score

def check_classification_run(run_path, set_path):
    # Output to return listing all errors
    output = ''
    # List of the provided videos
    list_of_videos_set = getListOfFiles(set_path)
    # XML filled or created by participant with each line being a video with its classification
    set_submission = set_path + '.xml'
    # Get the submission
    tree = ElementTree.parse(set_submission)
    root = tree.getroot()
    classes_submitted = {}
    for video in root:
        classes_submitted[video.get('name')] = video.get('class')

    if len(list_of_videos_set)!=len(classes_submitted):
        output += '\nNot the same number of videos in the set and in the submission'

    for video in list_of_videos_set:
        video_name = os.path.splitext(os.path.basename(video))[0]
        if video_name not in classes_submitted.keys():
            output += '\n%s not found in submission' % (video_name)
        elif classes_submitted[video_name] not in dict_of_moves:
            output += '\n%s: Class %s unknown for video %s' % (classes_submitted[video_name], video_name)

    if output == '':
        return 'OK'
    else:
        return output

def check_detection_run(run_path, set_path):
    original_xml_list = [f for f in os.listdir(set_path) if f.endswith('.xml')]
    submitted_xml_list = [f for f in os.listdir(run_path) if f.endswith('.xml')]
    output = ''

    if len(original_xml_list) != len(submitted_xml_list):
        output += '\nNot the same number of xml files'

    for file in original_xml_list:
        action_list = []
        video = cv2.VideoCapture(os.path.join(set_path,os.path.splitext(file)[0] + ".mp4"))

        try:
            N_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            N_frames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

        video.release()

        if file not in submitted_xml_list:
            output += '\n%s not found in submission' % (file)
            continue

        tree = ElementTree.parse(os.path.join(run_path, file))
        root = tree.getroot()
        for action in root:
            begin = int(action.get('begin'))
            end = int(action.get('end'))
            action_list.append([begin, end])
            if (end-begin<0) or (begin<0) or (end<0):
                output += '\n%s: An action must have boundaries making sense - begin %d and end %d given' % (file, begin, end)

            if end > N_frames or begin > N_frames:
                output += '\n%s: An action must have boundaries within the video - begin %d end %d given for video with %d frames.' % (file, begin, end, N_frames)
        
        # Compare each detected stroke one by one per video in order to check if the iou is less than .5 (commutative) - may take a while
        for idx, action1 in enumerate(action_list[:-1]):
            vector1 = np.zeros(N_frames)
            begin = action1[0]
            end = action1[1]
            if begin==end:
                vector1[begin] = 1
            else:
                vector1[begin:end] = 1
            for action2 in action_list[idx+1:]:
                vector2 = np.zeros(N_frames)
                vector2[action2[0]:action2[1]] = 1
                if compute_iou(vector1, vector2)>.5:
                    output += '\n%s: Two detected strokes should have an iou < .5 or else considred has a same stroke. (stroke1 [%d,%d] stroke2 [%d,%d]' % (file, action1[0], action1[1], action2[0], action2[1])

    if output == '':
        return 'OK'
    else:
        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Let\'s test your submitted files')
    parser.add_argument('path', help='Folder path in which you have the subfolders classificationTask and detectionTask containg subfolders representings your runs in which there are the xml files filled')
    parser.add_argument('set_path', nargs='?', default='test', help='Set on which the run has been done (per default test but you may run some checks on your validation and train sets too)')
    args = parser.parse_args()

    print(args.set_path)

    if args.set_path not in ['train', 'validation', 'test', 'testGT']:
        raise ValueError('Please provide a correct set name: train, validation or test. %s provided.' % (args.set_path))

    classification_path = os.path.join(args.path, "classificationTask")
    if os.path.isdir(classification_path):
        print('Classification task:')
        idx=0
        for idx, run in enumerate(os.listdir(classification_path)):
            run_path = os.path.join(classification_path, run)
            if os.path.isdir(run_path):
                idx+=1
                print('Run %d (%s): %s' % (idx, run, check_classification_run(run_path, os.path.join("classificationTask", args.set_path))))
            
    detection_path = os.path.join(args.path, "detectionTask")
    if os.path.isdir(detection_path):
        print('Detection task:')
        idx=0
        for idx, run in enumerate(os.listdir(detection_path)):
            run_path = os.path.join(detection_path, run)
            if os.path.isdir(run_path):
                idx+=1
                print('Run %d (%s): %s' % (idx, run, check_detection_run(run_path, os.path.join("detectionTask", args.set_path))))
    
    print('Test done')
    