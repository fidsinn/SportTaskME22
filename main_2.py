import os
import time
import datetime
import torch

from utils import *
from init_data import create_working_tree


# Classes

'''
Model variables
'''
class My_variables():
    def __init__(self, working_path, task_name, size_data=[320,180,96], model_load='2022-06-16_13-10-40', cuda=True, batch_size=10, workers=10, epochs=2000, lr=0.0001, nesterov=True, weight_decay=0.005, momentum=0.5):
        self.size_data = np.array(size_data)
        self.cuda = cuda
        self.workers = workers
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.lr_min = 0.000005
        self.lr_max = 0.01
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.momentum = momentum
        if model_load is None:
            self.model_name = os.path.join(working_path, 'Models', task_name, '%s' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
            self.train_model = True
        else:
            self.model_name = os.path.join(working_path, 'Models', task_name, model_load)
            self.train_model = False
        os.makedirs(self.model_name, exist_ok=True)
        if cuda:
            self.dtype = torch.cuda.FloatTensor
            os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '1'
        else:
            self.dtype = torch.FloatTensor
        self.log = setup_logger('model_log', os.path.join(self.model_name, 'model_%s.log' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))))

        with open(os.path.join(self.model_name, 'model_info.json'), 'w') as f:
            json.dump(str(self.__dict__.copy()), f, indent=4)


'''
My_stroke class used for encoding the annotations
'''
class My_stroke:
    def __init__(self, video_path, begin, end, move):
        self.video_path = video_path
        self.begin = begin
        self.end = end
        if type(move) is int:
            self.move = move
        elif move == 'Unknown':
            self.move = 0
        else:
            self.move = list_of_strokes.index(move)

    def my_print(self, log=None):
        print_and_log('Video : %s\tbegin : %d\tEnd : %d\tClass : %s' % (self.video_path, self.begin, self.end, self.move), log=log)

# Global vars

'''
According to overview paper
'''
list_of_strokes = [
    'Negative',
    
    'Serve Forehand Backspin',
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

# def get_classification_strokes(working_folder_task):
#     set_path = os.path.join(working_folder_task, 'train')
#     train_strokes = [My_stroke(os.path.join(set_path, action_class, f), 0, len(os.listdir(os.path.join(set_path, action_class, f))), action_class)
#         for action_class in os.listdir(set_path) for f in os.listdir(os.path.join(set_path, action_class))]
#     set_path = os.path.join(working_folder_task, 'validation')
#     validation_strokes = [My_stroke(os.path.join(set_path, action_class, f), 0, len(os.listdir(os.path.join(set_path, action_class, f))), action_class)
#         for action_class in os.listdir(set_path) for f in os.listdir(os.path.join(set_path, action_class))]
#     set_path = os.path.join(working_folder_task, 'test')
#     test_strokes = [My_stroke(os.path.join(set_path, f), 0, len(os.listdir(os.path.join(set_path, f))), 'Unknown') for f in os.listdir(set_path)]
#     return train_strokes, validation_strokes, test_strokes

# def classification_task(working_folder, data_in = 'rgb', log=None, test_strokes_segmentation=None):
#     '''
#     Main of the classification task
#     Perform also on the detection task when the videos for segmentation are provided
#     '''
#     print_and_log('\nClassification Task', log=log)
#     # Initialization
#     reset_training(1)
#     task_name = 'classificationTask'
#     task_path = os.path.join(working_folder, data_in, task_name)

#     # Split
#     train_strokes, validation_strokes, test_strokes = get_classification_strokes(task_path)

#     # Model variables
#     args = My_variables(working_folder, task_name)
    
#     ## Architecture with the output of the lenght of possible classes - (Unknown not counted)
#     model = make_architecture(args, len(list_of_strokes))

#     # Loaders
#     train_loader, validation_loader, test_loader = get_data_loaders(train_strokes, validation_strokes, test_strokes, args.size_data, args.batch_size, args.workers)

#     # Training process
#     if args.train_model:
#         train_model(model, args, train_loader, validation_loader)
    
#     # Test process
#     load_checkpoint(model, args)
#     test_model(model, args, test_loader, list_of_strokes=list_of_strokes)
#     test_prob_and_vote(model, args, test_strokes, list_of_strokes=list_of_strokes)
#     if test_strokes_segmentation is not None:
#         test_videos_segmentation(model, args, test_strokes_segmentation, sum_stroke_scores=True)
#     return 1



if __name__ == "__main__":
    # Chrono
    start_time = time.time()

    print('Working GPU device:',torch.cuda.get_device_name(torch.cuda.current_device()))

    # Mode s, srgb, rgb TODO: This will be kicked eventualy

    # MediaEval Task source folder
    source_folder = 'data'

    # Folder to save work
    # s      -> skeleton stream
    # rgb    -> rgb stream
    # srgb   -> srgb stream
    working_folder = 'working_folder'
    
    # Log file
    log_folder = os.path.join(working_folder, 'logs')
    os.makedirs(log_folder, exist_ok=True)
    log = setup_logger('my_log', os.path.join(log_folder, '%s.log' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))))
    
    # Prepare work tree (respect levels for correct extraction of the frames)
    create_working_tree(working_folder, source_folder, frame_width=320, log=log)
    print_and_log('Working tree created in %ds' % (time.time()-start_time), log=log)


    # Tasks
    # detection_task(working_folder, source_folder, log=log)
    # classification_task(working_folder, log=log, test_strokes_segmentation=get_videos_list(os.path.join(working_folder, 'detectionTask', 'test')))
    
    print_and_log('All Done in %ds' % (time.time()-start_time), log=log)