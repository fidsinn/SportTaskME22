'''Evaluation script by Pierre-Etienne MARTIN dedicated to the Sport Task for MediaEval 2021'''
import matplotlib
# To be able to save figure using screen with matplotlib
matplotlib.use('Agg')
import os
import cv2
import argparse
import numpy as np
from xml.etree import ElementTree
import operator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# List of stroke and corresponding super class - used fpr confusion matrices
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

dict_of_strokes_hand = { 'Serve Forehand Backspin':'Forehand',
                               'Serve Forehand Loop':'Forehand',
                               'Serve Forehand Sidespin':'Forehand',
                               'Serve Forehand Topspin':'Forehand',

                               'Serve Backhand Backspin':'Backhand',
                               'Serve Backhand Loop':'Backhand',
                               'Serve Backhand Sidespin':'Backhand',
                               'Serve Backhand Topspin':'Backhand',

                               'Offensive Forehand Hit':'Forehand',
                               'Offensive Forehand Loop':'Forehand',
                               'Offensive Forehand Flip':'Forehand',

                               'Offensive Backhand Hit':'Backhand',
                               'Offensive Backhand Loop':'Backhand',
                               'Offensive Backhand Flip':'Backhand',

                               'Defensive Forehand Push':'Forehand',
                               'Defensive Forehand Block':'Forehand',
                               'Defensive Forehand Backspin':'Forehand',

                               'Defensive Backhand Push':'Backhand',
                               'Defensive Backhand Block':'Backhand',
                               'Defensive Backhand Backspin':'Backhand'}

list_of_strokes_hand = ['Forehand', 'Backhand']

dict_of_strokes_serve = { 'Serve Forehand Backspin':'Serve',
                               'Serve Forehand Loop':'Serve',
                               'Serve Forehand Sidespin':'Serve',
                               'Serve Forehand Topspin':'Serve',

                               'Serve Backhand Backspin':'Serve',
                               'Serve Backhand Loop':'Serve',
                               'Serve Backhand Sidespin':'Serve',
                               'Serve Backhand Topspin':'Serve',

                               'Offensive Forehand Hit':'Offensive',
                               'Offensive Forehand Loop':'Offensive',
                               'Offensive Forehand Flip':'Offensive',

                               'Offensive Backhand Hit':'Offensive',
                               'Offensive Backhand Loop':'Offensive',
                               'Offensive Backhand Flip':'Offensive',

                               'Defensive Forehand Push':'Defensive',
                               'Defensive Forehand Block':'Defensive',
                               'Defensive Forehand Backspin':'Defensive',

                               'Defensive Backhand Push':'Defensive',
                               'Defensive Backhand Block':'Defensive',
                               'Defensive Backhand Backspin':'Defensive',

                               'Negative':'Negative'}

list_of_strokes_serve = ['Serve', 'Offensive', 'Defensive']


dict_of_strokes_serve_hand = { 'Serve Forehand Backspin':'Serve Forehand',
                               'Serve Forehand Loop':'Serve Forehand',
                               'Serve Forehand Sidespin':'Serve Forehand',
                               'Serve Forehand Topspin':'Serve Forehand',

                               'Serve Backhand Backspin':'Serve Backhand',
                               'Serve Backhand Loop':'Serve Backhand',
                               'Serve Backhand Sidespin':'Serve Backhand',
                               'Serve Backhand Topspin':'Serve Backhand',

                               'Offensive Forehand Hit':'Offensive Forehand',
                               'Offensive Forehand Loop':'Offensive Forehand',
                               'Offensive Forehand Flip':'Offensive Forehand',

                               'Offensive Backhand Hit':'Offensive Backhand',
                               'Offensive Backhand Loop':'Offensive Backhand',
                               'Offensive Backhand Flip':'Offensive Backhand',

                               'Defensive Forehand Push':'Defensive Forehand',
                               'Defensive Forehand Block':'Defensive Forehand',
                               'Defensive Forehand Backspin':'Defensive Forehand',

                               'Defensive Backhand Push':'Defensive Backhand',
                               'Defensive Backhand Block':'Defensive Backhand',
                               'Defensive Backhand Backspin':'Defensive Backhand'}

list_of_strokes_serve_hand = ['Serve Forehand', 'Serve Backhand', 'Offensive Forehand', 'Offensive Backhand', 'Defensive Forehand', 'Defensive Backhand']

'''
Function to create the confusion matrix
'''
def plot_confusion_matrix(cm, classes, save_path, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    
    acc = np.mean(np.array([cm[i,i] for i in range(len(cm))]).sum()/cm.sum()) * 100
    cm = cm / [max(tmp,1) for tmp in cm.sum(axis=1)]
    acc_2 = np.array([cm[i,i] for i in range(len(cm))])

    title = 'Accuracy of %.1f%%\n$\\mu$ = %.1f with $\\sigma$ = %.1f' % (acc, np.mean(acc_2)*100, np.std(acc_2)*100)
    if len(classes)>=12:
        plt.subplots(figsize=(12,12))
    elif len(classes)>=6:
        plt.subplots(figsize=(8,8))
    else:
        plt.subplots(figsize=(5,5))

    plt.imshow(cm.astype('float'), interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title, fontsize=16)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)

    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

def compute_iou(gt, prediction, print_option=False):
    gt_not = list(map(operator.not_, gt))
    intersection = np.logical_and(gt, prediction)
    union = np.logical_or(gt, prediction)
    iou_score = 1.*np.sum(intersection) / np.sum(union)
    
    if print_option:
        prediction_not = list(map(operator.not_, prediction))
        intersection_not = np.logical_and(gt_not, prediction_not)
        TP = np.sum(intersection)
        TN = np.sum(intersection_not)
        FP = np.sum(np.logical_and(prediction, gt_not))
        FN = np.sum(np.logical_and(prediction_not, gt))
        print('Iou is : %.3f   TP: %d, TN: %d, FP: %d, FN: %d' % (iou_score, TP, TN, FP, FN))
        print('\trecall: %.3f, Precision: %.3f, TNR: %.3f, FPR: %.3f, FNR: %.3f' % (TP/(TP+FN), TP/(TP+FP), TN/(TN+FP), FP/(FP+TN), FN/(TP+FN)))
    return iou_score

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

def evaluate_classification(run_path, set_path):
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
        raise ValueError('\nNot the same number of videos in the set and in the submission.')

    # Get the gt
    classes_gt = {}
    if set_path.endswith('test'):
        # Specific path for the organizers
        gt_path = 'private/classificationTask/text.xml'
        if not os.path.exists(gt_path):
            raise ValueError('The gt path %s does not exist.\nOnly the organizers have the gt test xml.' % (gt_path))
        tree = ElementTree.parse(gt_path)
        root = tree.getroot()
        classes_submitted = {}
        for video in root:
            classes_submitted[video.get('name')] = video.get('class')
    else:
        for video in list_of_videos_set:
            video_name = os.path.splitext(os.path.basename(video))[0]
            video_class_gt = video.split('/')[-2]
            classes_gt[video_name] = video_class_gt    

    # For the confusion matrix we store the ground truth and the predictions
    gt_conf_matrix = []
    prediction_conf_matrix = []
    numCorrectActions_h = dict()
    numActions_h = dict()
    for move in dict_of_moves:
        numCorrectActions_h[move] = 0
        numActions_h[move] = 0

    for video_name in classes_gt.keys():
        if video_name not in classes_submitted.keys():
            raise ValueError('\n%s not found in submission' % (video_name))
        elif classes_submitted[video_name] not in dict_of_moves:
            raise ValueError('\n%s: Class %s unknown for video %s' % (classes_submitted[video_name], video_name))
        class_gt = classes_gt[video_name]
        class_submitted = classes_submitted[video_name]

        # For Conf Matrix and stats
        gt_conf_matrix.append(class_gt)
        prediction_conf_matrix.append(class_submitted)

        numActions_h[class_gt] += 1
        numActions += 1
        numActions_h[classes_gt[video_name]] += 1

        if class_gt == class_submitted:
            numCorrectActions_h[class_gt] += 1
            numCorrectActions += 1

    # Save different confusions matrices
    plot_confusion_matrix(
        confusion_matrix(gt_conf_matrix, prediction_conf_matrix, labels=dict_of_moves),
        dict_of_moves,
        os.path.join(run_path, 'cm.png'))
    plot_confusion_matrix(
        confusion_matrix([dict_of_strokes_serve_hand[i] for i in gt_conf_matrix], [dict_of_strokes_serve_hand[i] for i in prediction_conf_matrix], labels=list_of_strokes_serve_hand),
        list_of_strokes_serve_hand,
        os.path.join(run_path, 'cm_serve_hand.png'))
    plot_confusion_matrix(
        confusion_matrix([dict_of_strokes_hand[i] for i in gt_conf_matrix], [dict_of_strokes_hand[i] for i in prediction_conf_matrix], labels=list_of_strokes_hand),
        list_of_strokes_hand,
        os.path.join(run_path, 'cm_hand.png'))
    plot_confusion_matrix(
        confusion_matrix([dict_of_strokes_serve[i] for i in gt_conf_matrix], [dict_of_strokes_serve[i] for i in prediction_conf_matrix], labels=list_of_strokes_serve),
        list_of_strokes_serve,
        os.path.join(run_path, 'cm_serve.png'))
                
    accuracy = numCorrectActions / float(numActions)
    print('\nGlobal accuracy={}/{}={}\n'.format(numCorrectActions, numActions, accuracy))
    # print('Accuracy per move class:')
    # for move in dict_of_moves:
    #     if (numActions_h[move] != 0):
    #         print(' {} : accuracy={}/{}={}'.format(move, numCorrectActions_h[move], numActions_h[move], numCorrectActions_h[move]/numActions_h[move]))
    #     else:
    #         print(' {} : accuracy=N/A'.format(move))

def evaluate_detection(run_path, set_path):
    original_xml_list = [f for f in os.listdir(set_path) if f.endswith('.xml')]

    num_gt_actions_total = 0
    num_submitted_actions_total = 0
    gt_array_total = []
    submitted_array_total = []
    iou_thresholds = [0.5 + idx/20 for idx in range(10)]
    TP = np.zeros(len(iou_thresholds))
    FP = np.zeros(len(iou_thresholds))
    FN = np.zeros(len(iou_thresholds))
    
    for file in original_xml_list:
        action_list_gt = []
        action_list_pred = []
        gt_file_path = os.path.join(set_path, file)
        submitted_file_path = os.path.join(run_path, file)

        video = cv2.VideoCapture(os.path.join(set_path,os.path.splitext(file)[0] + ".mp4"))

        try:
            N_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            N_frames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        
        video.release()
        gt_array = np.zeros(N_frames)
        submitted_array = np.zeros(N_frames)
        num_gt_actions = 0
        num_submitted_actions = 0

        if not os.path.exists(submitted_file_path):
            raise ValueError('The result xml file does not exist: %s' % (submitted_file_path))
        
        tree = ElementTree.parse(gt_file_path)
        root = tree.getroot()
        for action in root:
            num_gt_actions += 1
            begin = int(action.get('begin'))
            end = int(action.get('end'))
            action_list_gt.append([begin, end])
            gt_array[begin:end] = 1

        tree = ElementTree.parse(submitted_file_path)
        root = tree.getroot()
        for action in root:
            num_submitted_actions += 1
            begin = int(action.get('begin'))
            end = int(action.get('end'))
            action_list_pred.append([begin, end])
            if (end-begin<0) or (begin<0) or (end<0):
                raise ValueError('%s: An action must have boundaries making sense - begin %d and end %d given' % (file, begin, end))
            if (end > N_frames) or (begin > N_frames):
                raise ValueError('%s: An action must have boundaries within the video - begin %d end %d given for video with %d frames.' % (file, begin, end, N_frames))
            submitted_array[begin:end] = 1

        # Compare each detected stroke with gt per video in order to get iou and accordingly increment TP, FP, FN
        check_FN = np.array([[1 for _ in iou_thresholds] for action in action_list_gt])
        for idx_pred, action_pred in enumerate(action_list_pred):
            check_FP = [1 for _ in iou_thresholds]
            vector_pred = np.zeros(N_frames)
            vector_pred[action_pred[0]:action_pred[1]] = 1
            for idx_gt, action_gt in enumerate(action_list_gt):
                vector_gt = np.zeros(N_frames)
                vector_gt[action_gt[0]:action_gt[1]] = 1
                iou = compute_iou(vector_pred, vector_gt)
                for idx_th, iou_th in enumerate(iou_thresholds):
                    # Stroke has been for the first time detected and overlaps with GT stroke
                    if iou >= iou_th and check_FN[idx_gt,idx_th]!=0:
                        TP[idx_th]+=1
                        check_FP[idx_th]=0
                        check_FN[idx_gt,idx_th]=0
            # Stroke has been detected but no GT Stroke
            FP+=check_FP
        # GT stroke has not been detected
        FN+=check_FN.sum(0)
  
        gt_array_total = np.append(gt_array_total, gt_array)
        submitted_array_total = np.append(submitted_array_total, submitted_array)
        num_gt_actions_total += num_gt_actions
        num_submitted_actions_total += num_submitted_actions

    print("\nFrame as one object to detect:\n")
    # print("Number of actions detected vs GT: %d / %d" % (num_submitted_actions_total, num_gt_actions_total))
    compute_iou(gt_array_total, submitted_array_total, print_option=True)

    for idx in range(len(iou_thresholds)):
        if num_gt_actions_total != TP[idx]+FN[idx]:
            print('This should not appear. Contact the organizers.')

    print('\nStroke as one object to detect:\n')
    # Compute recall and precision (graph possible)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)

    # # Max left
    # recall_sorted, precision_sorted = zip(*sorted(zip(recall, precision)))
    # recall_sorted = np.array(recall_sorted)
    # precision_sorted_maxleft = [max(precision_sorted[idx:]) for idx in range(len(precision_sorted))]
    # AP = (precision_sorted_maxleft*(np.append(recall_sorted[0],recall_sorted[1:]-recall_sorted[:-1]))).sum()

    for idx in [0,5]:
        print("With IoU threshold of %g" % iou_thresholds[idx])
        print('\tPrecision: %f, Recall: %f, dedicated AP: %f' % (precision[idx], recall[idx], precision[idx]*recall[idx]))
    print("\nMean Average Precision at IoU=.50:.05:.95 = %f" % np.mean(precision*recall))

    return 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Evaluation of the participant')
    parser.add_argument('--path', '-p', help='Folder path in which you have the subfolders classificationTask and detectionTask containg subfolders representings the runs in which there are the xml files filled')
    parser.add_argument('--set_path', '-sp', nargs='?', default='testGT', help='Set on which the run has been done (per default test but you may run some checks on your validation and train sets too)')
    args = parser.parse_args()

    if args.set_path not in ['train', 'validation', 'test', 'testGT']:
        raise ValueError('Please provide a correct set name (train, validation or test')

    classification_path = os.path.join(args.path, "classificationTask")
    if os.path.isdir(classification_path):
        print('\nClassification task:')
        idx=0
        for idx, run in enumerate(os.listdir(classification_path)):
            run_path = os.path.join(classification_path, run)
            if os.path.isdir(run_path):
                idx+=1
                print('\nRun %d (%s):' % (idx, run))
                try:
                    evaluate_classification(run_path, os.path.join("classificationTask", args.set_path))
                except ValueError as error:
                    print(error)
                    continue

            
    detection_path = os.path.join(args.path, "detectionTask")
    if os.path.isdir(detection_path):
        print('\nDetection task:')
        idx=0
        for idx, run in enumerate(os.listdir(detection_path)):
            run_path = os.path.join(detection_path, run)
            if os.path.isdir(run_path):
                idx+=1
                print('\nRun %d (%s)' % (idx, run))
                try:
                    evaluate_detection(run_path, os.path.join("detectionTask", args.set_path))
                except ValueError as error:
                    print(error)
                    continue
    
    print('Evaluation done')