import os
import time
import datetime
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import json
import xml.etree.ElementTree as ET

from utils import *
from model import *
from init_data import create_working_tree

import argparse
import sys

print('Python version : ', platform.python_version())
print('OpenCV version  : ', cv2.__version__)
print('Torch version : ', torch.__version__)
# Opencv use several cpus by default for simple operation. Using only one allows loading data in parallel much faster
cv2.setNumThreads(0)
print('Nb of threads for OpenCV : ', cv2.getNumThreads())



# Classes

'''
Model variables
'''
class My_variables():
    def __init__(self, working_path, data_in, task_name, epochs, model_load, size_data=[320,180,96], cuda=True, batch_size=10, workers=10, lr=0.0001, nesterov=True, weight_decay=0.005, momentum=0.5):
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
        self.data_in = data_in 
        if model_load is None:
            self.model_name = os.path.join(working_path, 'Models', task_name + '_' + data_in[0] + '-' + data_in[1], '%s' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
            self.train_model = True
        else:
            self.model_name = os.path.join(working_path, 'Models', task_name + '_' + data_in[0] + '-' + data_in[1], model_load)
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
            self.move = LIST_OF_STROKES.index(move)

    def my_print(self, log=None):
        print_and_log('Video : %s\tbegin : %d\tEnd : %d\tClass : %s' % (self.video_path, self.begin, self.end, self.move), log=log)

'''
My_dataset class which uses My_stroke class to be used in the data loader
'''
class My_dataset(Dataset):
    def __init__(self, dataset_list, size_data, augmentation=False):
        self.dataset_list_1 = dataset_list[0]
        self.dataset_list_2 = dataset_list[1]
        self.size_data = size_data
        self.augmentation = augmentation

    def __len__(self):
        return len(self.dataset_list_1)

    def __getitem__(self, idx):
        frames_1, frames_2 = get_data(self.dataset_list_1[idx], self.dataset_list_2[idx], self.size_data, self.augmentation)
        sample = {'stream_1': torch.FloatTensor(frames_1), 'stream_2': torch.FloatTensor(frames_2), 'label' : self.dataset_list_1[idx].move, 'my_stroke' : {'video_path':self.dataset_list_1[idx].video_path, 'begin':self.dataset_list_1[idx].begin, 'end':self.dataset_list_1[idx].end}}
        return sample



# Global vars
'''
According to overview paper
'''
LIST_OF_STROKES = [
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

'''Infer Negative Samples from annotation betwen strokes when there are more than length_min frames'''
def build_negative_strokes(stroke_list, length_min=200):
    video_path = ''
    for stroke in stroke_list.copy():
        if stroke.video_path != video_path:
            video_path = stroke.video_path
            begin_negative = 0
        end_negative = stroke.begin
        for end in range(begin_negative+length_min, end_negative, length_min):
            stroke_list.append(My_stroke(video_path, end-length_min, end, 0))
        begin_negative = stroke.end


# Get random augmentation values (for all frames in range from begin to end?)
def get_augmentation_parameters(begin, end, size_data):
    '''
    Get augmentation parameters
    '''
    angle = (random.random()* 2 - 1) * 10
    zoom = 1 + (random.random()* 2 - 1) * 0.1
    tx = random.randint(-0.1 * size_data[0], 0.1 * size_data[0])
    ty = random.randint(-0.1 * size_data[1], 0.1 * size_data[1])
    flip = random.randint(0,1)

    # Normal distribution to pick where to begin #
    mu = begin + (end + 1 - begin - size_data[2])/2
    sigma = (end + 1 - begin - size_data[2])/6
    begin = -1

    if sigma <= 0:
        begin = int(mu)
    else:
        count=0
        while not begin <= begin <= end + 1 - size_data[2]:
            begin = int(np.random.normal(mu, sigma))
            count+=1
            if count>100:
                raise ValueError('Warning: augmentation with picking frame has a problem\n mu %d\nbegin %d\nend %d\nsigma %d' % (mu, begin, end, sigma))
    return angle, zoom, tx, ty, flip, max(begin,0)


def apply_augmentation(data, zoom, R_matrix, flip):
    '''
    Apply zoo, rotation (cneter with some translation) and flip to the data
    '''
    data = cv2.resize(cv2.warpAffine(data, R_matrix, (data.shape[1], data.shape[0])), (0,0), fx = zoom, fy = zoom)
    # Flip
    if flip:
        data = cv2.flip(data, 1)
    return data


#  get data and aplay some augmentaion if specified
def get_data(data_1, data_2, size_data, augmentation):
    frame_data_1 = []
    frame_data_2 = []
    begin = data_1.begin
    end = data_1.end
    if augmentation:
        angle, zoom, tx, ty, flip, begin = get_augmentation_parameters(begin, end, size_data)
        R_matrix = cv2.getRotationMatrix2D((size_data[0]//2-tx, size_data[1]//2-ty), angle, 1)
    else:
        angle, zoom, tx, ty, flip, begin = 0, 1, 0, 0, 0, max((begin+end-size_data[2])//2,0)
    
    max_frame_number = len(os.listdir(data_1.video_path))-1

    for frame_number in range(begin, begin + size_data[2]):
        if frame_number > max_frame_number:
            frame_number = max_frame_number
        try:
            frame_1 = cv2.imread(os.path.join(data_1.video_path, '%08d.png' % frame_number))
            frame_1 = cv2.resize(frame_1, (size_data[0], size_data[1])).astype(float) / 255


            frame_2 = cv2.imread(os.path.join(data_2.video_path, '%08d.png' % frame_number))
            frame_2 = cv2.resize(frame_2, (size_data[0], size_data[1])).astype(float) / 255


            if augmentation:
                frame_1 = apply_augmentation(frame_1, zoom, R_matrix, flip)
                frame_2 = apply_augmentation(frame_2, zoom, R_matrix, flip)
        except:
            raise ValueError('Problem with %s or %s begin %d size %s' % (os.path.join(data_1.video_path, '%08d.png' % frame_number),os.path.join(data_2.video_path, '%08d.png' % frame_number), begin, str(size_data)))
        frame_data_1.append(cv2.split(frame_1))
        frame_data_2.append(cv2.split(frame_2))
    # To match size_data variable, transposition is needed. From (T,C,H,W) to (C,W,H,T).
    frame_data_1 = np.transpose(frame_data_1, (1, 3, 2, 0))
    frame_data_2 = np.transpose(frame_data_2, (1, 3, 2, 0))
    return frame_data_1, frame_data_2




'''
Build dataloader from list of strokes
'''
def get_data_loaders(train_strokes, validation_strokes, test_strokes, size_data, batch_size, workers):
    # Build Dataset
    train_set = My_dataset(train_strokes, size_data)
    validation_set = My_dataset(validation_strokes, size_data)
    test_set = My_dataset(test_strokes, size_data)

    # Loaders of the Datasets
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=workers, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=workers, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=workers)
    return train_loader, validation_loader, test_loader



'''
Model Architecture
'''
def make_architecture(args, output_size):
    print_and_log('Make Model', log=args.log)
    model = CCNAttentionNet_TwoStream_V1(args.size_data.copy(), output_size)
    print_and_log('Model %s created' % (model.__class__.__name__), log=args.log)
    ## Use GPU
    if args.cuda:
        model.cuda()
    return model

def save_checkpoint(args, model, optimizer, epoch, val_loss):
    torch.save({'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss}, os.path.join(args.model_name, 'checkpoint.pth'))
    print_and_log('Model %s saved' % (args.model_name), log=args.log)
    return 1

def load_checkpoint(model, args, optimizer=None):
    checkpoint = torch.load(os.path.join(args.model_name, 'checkpoint.pth'))
    model.load_state_dict(checkpoint['model_state_dict']) #, map_location=lambda storage, loc: storage
    # model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # val_loss = checkpoint['val_loss']
    print_and_log('Model %s loaded' % (args.model_name), log=args.log)
    return 1

def change_lr(optimizer, args, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        args.lr = lr
        print_and_log('Learning rate changed to %g' % lr, log=args.log)


'''
Training is split in train epoch and validation epoch and produce a plot
'''
def train_model(model, args, train_loader, validation_loader):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    # Chrono
    start_time = time.time()
    print_and_log('\nTraining...', log=args.log)

    # For plot
    loss_train = []
    loss_val = []
    acc_val = []
    acc_train = []
    min_loss_val = 1000
    max_acc = 0
    wait_change_lr = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        # Train and validation step and save loss and acc for plot
        loss_train_, acc_train_ = train_epoch(epoch, args, model, train_loader, optimizer, criterion)
        loss_val_, acc_val_ = validation_epoch(epoch, args, model, validation_loader, criterion)

        loss_train.append(loss_train_)
        acc_train.append(acc_train_)
        loss_val.append(loss_val_)
        acc_val.append(acc_val_)
        wait_change_lr += 1

        # Best model saved accoridng to loss
        if loss_val_ < min_loss_val:
            save_checkpoint(args, model, optimizer, epoch, loss_val)
            min_loss_val = loss_val_
            max_acc = acc_val_
            best_epoch = epoch
            wait_change_lr = 0

        # Change lr according to evolution of the loss
        if wait_change_lr > 30:
            if .99*np.mean(loss_train[-30:-10]) < np.mean(loss_train[-10:]):
                print_and_log("Diff Loss : %g" % (np.mean(loss_train[-30:-10])-np.mean(loss_train[-10:])), log=args.log)
                load_checkpoint(model, args, optimizer)
                if args.lr < args.lr_min:
                    change_lr(optimizer, args, args.lr_max)
                else:
                    change_lr(optimizer, args, args.lr/5)
                wait_change_lr = 0

    print_and_log('Best model obtained with acc of %.3g, loss of %.3g at epoch %d - time for training: %ds' % (max_acc, min_loss_val, best_epoch, int(time.time()-start_time)), log=args.log)
    make_train_figure(loss_train, loss_val, acc_val, acc_train, os.path.join(args.model_name, 'Train.png'))
    return 1

'''
Update of the model in one epoch
'''
def train_epoch(epoch, args, model, data_loader, optimizer, criterion):
    model.train()
    pid = os.getpid()
    N = len(data_loader.dataset)
    begin_time = time.time()
    aLoss = 0
    Acc = 0

    for batch_idx, batch in enumerate(data_loader):
        # Get batch tensor
        stream_1, stream_2, label = batch['stream_1'], batch['stream_2'], batch['label']

        stream_1 = Variable(stream_1.type(args.dtype))
        stream_2 = Variable(stream_2.type(args.dtype))
        label = Variable(label.type(args.dtype).long())

        optimizer.zero_grad()
        output = model(stream_1, stream_2)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        aLoss += loss.item()
        Acc += output.data.max(1)[1].eq(label.data).cpu().sum().numpy()
        progress_bar((batch_idx + 1) * args.batch_size, N, '%d Training - Epoch : %d - Batch Loss = %.5g' % (pid, epoch, loss.item()))

    aLoss /= N
    progress_bar(N, N, 'Train - Epoch %d - Loss = %.5g - Accuracy = %.3g (%d/%d) - Time = %ds' % (epoch, aLoss, Acc/N, Acc, N, time.time() - begin_time), 1, log=args.log)
    return aLoss, Acc/N


'''
Validation of the model in one epoch
'''
def validation_epoch(epoch, args, model, data_loader, criterion):
    with torch.no_grad():
        model.eval() # Set model to evaluation mode - needed for batchnorm
        begin_time = time.time()
        pid = os.getpid()
        N = len(data_loader.dataset)
        _loss = 0
        _acc = 0

        for batch_idx, batch in enumerate(data_loader):
            progress_bar(batch_idx*args.batch_size, N, '%d - Validation' % (pid))
            
            stream_1, stream_2, label = batch['stream_1'], batch['stream_2'], batch['label']
            stream_1 = Variable(stream_1.type(args.dtype))
            stream_2 = Variable(stream_2.type(args.dtype))
            label = Variable(label.type(args.dtype).long())

            output = model(stream_1, stream_2)
            _loss += criterion(output, label).item()
            _acc += output.data.max(1)[1].eq(label.data).cpu().sum().numpy()

        _loss /= N
        progress_bar(N, N, 'Validation - Loss = %.5g - Accuracy = %.3g (%d/%d) - Time = %ds' % (_loss, _acc/N, _acc, N, time.time() - begin_time), 1, log=args.log)
        return _loss, _acc/N

'''
Test Process
'''
def store_xml_data(my_stroke_list, predicted, xml_files, list_of_strokes=None):
    '''
    This function stores data to xml files from the list of stroke with predicted class.
    For detection it is saved when index predicted to 1
    '''
    for video_path, begin, end, prediction_index in zip(my_stroke_list['video_path'], my_stroke_list['begin'].tolist(), my_stroke_list['end'].tolist(), predicted):
        video_name = video_path.split('/')[-1]
        if list_of_strokes is None:
            if video_name not in xml_files:
                xml_files[video_name] = ET.Element('video')
            if prediction_index:
                stroke_xml = ET.SubElement(xml_files[video_name], 'action')
                stroke_xml.set('begin', str(begin))
                stroke_xml.set('end', str(end))
        else:
            if 'test' not in xml_files:
                xml_files['test'] = ET.Element('videos')
            stroke_xml = ET.SubElement(xml_files['test'], 'video')
            stroke_xml.set('name', '%s.mp4' % (video_name))
            stroke_xml.set('class', list_of_strokes[prediction_index])

def store_stroke_to_xml(my_stroke_list, xml_files, list_of_strokes=None):
    '''
    This function stores strokes in xml files.
    '''
    for my_stroke in my_stroke_list:
        video_name = my_stroke.video_path.split('/')[-1]
        if list_of_strokes is None:
            if video_name not in xml_files:
                xml_files[video_name] = ET.Element('video')
            if my_stroke.move:
                stroke_xml = ET.SubElement(xml_files[video_name], 'action')
                stroke_xml.set('begin', str(my_stroke.begin))
                stroke_xml.set('end', str(my_stroke.end))
        else:
            if 'test' not in xml_files:
                xml_files['test'] = ET.Element('videos')
            stroke_xml = ET.SubElement(xml_files['test'], 'video')
            stroke_xml.set('name', '%s.mp4' % (video_name))
            stroke_xml.set('class', list_of_strokes[my_stroke.move])

'''
Save the predictions in xml files
'''
def save_xml_data(xml_files, path_xml_save):
    for xml_name in xml_files:
        xml_file = open('%s.xml' % os.path.join(path_xml_save, xml_name), 'wb')
        xml_file.write(ET.tostring(xml_files[xml_name]))
        xml_file.close()

'''
Inference on test set
'''
def test_model(model, args, data_loader, list_of_strokes=None):
    with torch.no_grad():
        model.eval() # Set model to evaluation mode - needed for batchnorm
        xml_files = {}
        path_xml_save = os.path.join(args.model_name, 'xml_test')
        os.mkdir(path_xml_save)
        N = len(data_loader.dataset)
        
        for batch_idx, batch in enumerate(data_loader):
            # Get batch tensor
            stream_1, stream_2, my_stroke_list = batch['stream_1'], batch['stream_2'], batch['my_stroke']
            progress_bar(args.batch_size*batch_idx, N, 'Testing')
            stream_1 = Variable(stream_1.type(args.dtype))
            stream_2 = Variable(stream_2.type(args.dtype))
            output = model(stream_1, stream_2)
            _, predicted = torch.max(output.detach(), 1)
            store_xml_data(my_stroke_list, predicted, xml_files, list_of_strokes=list_of_strokes)

        progress_bar(N, N, 'Test done', 1, log=args.log)
        save_xml_data(xml_files, path_xml_save)

# Classification Task
# Get Stroke labels from the directory structure and save it in the My_stroke class
def get_classification_strokes(working_folder_task):
    set_path = os.path.join(working_folder_task, 'train')
    train_strokes = [My_stroke(os.path.join(set_path, action_class, f), 0, len(os.listdir(os.path.join(set_path, action_class, f))), action_class)
        for action_class in os.listdir(set_path) for f in os.listdir(os.path.join(set_path, action_class))]
    set_path = os.path.join(working_folder_task, 'validation')
    validation_strokes = [My_stroke(os.path.join(set_path, action_class, f), 0, len(os.listdir(os.path.join(set_path, action_class, f))), action_class)
        for action_class in os.listdir(set_path) for f in os.listdir(os.path.join(set_path, action_class))]
    set_path = os.path.join(working_folder_task, 'test')
    test_strokes = [My_stroke(os.path.join(set_path, f), 0, len(os.listdir(os.path.join(set_path, f))), 'Unknown') for f in os.listdir(set_path)]
    
    # they need to be sorted as list dir gets them in different orderes
    train_strokes.sort(key=lambda x: x.video_path, reverse=True)
    validation_strokes.sort(key=lambda x: x.video_path, reverse=True)
    test_strokes.sort(key=lambda x: x.video_path, reverse=True)

    return train_strokes, validation_strokes, test_strokes

def classification_task(working_folder, data_in, epochs, model_load, test_strokes_segmentation, log):
    '''
    Main of the classification task
    Perform also on the detection task when the videos for segmentation are provided
    '''
    print_and_log('\nClassification Task', log=log)
    # Initialization
    reset_training(1)
    task_name = 'classificationTask' 
    task_paths= []
    for data in data_in:
        task_paths.append(os.path.join(working_folder, data, task_name))

    # Split
    train_strokes_list, validation_strokes_list, test_strokes_list = [], [], []
    for path in task_paths:
        train_strokes, validation_strokes, test_strokes = get_classification_strokes(path)
        train_strokes_list.append(train_strokes)
        validation_strokes_list.append(validation_strokes)
        test_strokes_list.append(test_strokes)

    # Model variables
    args = My_variables(working_folder, data_in, task_name, epochs, model_load)
    
    # Architecture with the output of the lenght of possible classes - (Unknown not counted)
    # make two identical models
    model = make_architecture(args, len(LIST_OF_STROKES))

    # Loaders
    train_loader, validation_loader, test_loader = get_data_loaders(train_strokes_list, validation_strokes_list, test_strokes_list, args.size_data, args.batch_size, args.workers)

    # Training process
    if args.train_model:
        train_model(model, args, train_loader, validation_loader)
    
    # Test process
    load_checkpoint(model, args)
    # TODO make testing stuff work
    # test_model(model, args, test_loader, list_of_strokes=LIST_OF_STROKES)
    # test_prob_and_vote(model, args, test_strokes, list_of_strokes=LIST_OF_STROKES)
    # if test_strokes_segmentation is not None:
    #     test_videos_segmentation(model, args, test_strokes_segmentation, sum_stroke_scores=True)
    return 1

'''
Detection task
'''
def get_videos_list(data_folder):
    '''
    Get list of videos and transform it to strokes for segmentation purpose
    '''
    video_list = []
    for video in os.listdir(data_folder):
        video_path = os.path.join(data_folder, video)
        video_list.append(My_stroke(video_path, 0, len(os.listdir(video_path)), 0))
    return video_list

def get_annotations(xml_path, data_folder):
    '''
    Get annotations from xml files located in one folder and produce a list of My_stroke
    '''
    xml_list = [os.path.join(xml_path, f) for f in os.listdir(xml_path) if os.path.isfile(os.path.join(xml_path, f)) and f.split('.')[-1]=='xml']
    strokes_list = []
    for xml_file in xml_list:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        video_path = os.path.join(data_folder, xml_file.split('/')[-1].split('.')[0])
        for action in root:
            strokes_list.append(My_stroke(video_path, int(action.get('begin')), int(action.get('end')), 1))
        # Case of the test set in segmentation task - build proposals of size 150
        if len(root)==0: 
            for begin in range(0,len(os.listdir(video_path))-150,150):
                strokes_list.append(My_stroke(video_path, begin, begin+150, 0))
    build_negative_strokes(strokes_list)
    return strokes_list

def get_lists_annotations(task_source, task_path):
    '''
    Get the split of annotation and construct negative samples
    '''
    train_strokes = get_annotations(os.path.join(task_source, 'train'), os.path.join(task_path, 'train'))
    validation_strokes = get_annotations(os.path.join(task_source, 'validation'), os.path.join(task_path, 'validation'))
    test_strokes = get_annotations(os.path.join(task_source, 'test'), os.path.join(task_path, 'test'))
    return train_strokes, validation_strokes, test_strokes

def detection_task(working_folder, source_folder, data_in, epochs, model_load, log=None):
    '''
    Main of the detection task
    Return test segmentation video to try with the classification model
    '''
    print_and_log('\nDetection Task', log=log)
    # Initialization
    reset_training(1)
    task_name = 'detectionTask'
    task_paths = []

    task_source = os.path.join(source_folder, task_name)
    print('  task_source:', task_source)
    for data in data_in:
        task_path = os.path.join(working_folder, data, task_name)
        task_paths.append(task_path)

    # Split
    train_strokes_list, validation_strokes_list, test_strokes_list = [], [], []
    for path in task_paths:
        train_strokes, validation_strokes, test_strokes = get_lists_annotations(task_source, path)
        train_strokes_list.append(train_strokes)
        validation_strokes_list.append()
        test_strokes_list.append(test_strokes)
    print('  task_paths:', *task_path)
    print()


    # Model variables
    args = My_variables(working_folder, data_in, task_name, epochs, model_load)

    # Architecture with the output of the lenght of possible classes - Positive and Negative
    model = make_architecture(args, 2)

    # Loaders
    train_loader, validation_loader, test_loader = get_data_loaders(train_strokes_list, validation_strokes_list, test_strokes_list, args.size_data, args.batch_size, args.workers)

    # Training process
    if args.train_model:
        train_model(model, args, train_loader, validation_loader)
    
    # Test process
    load_checkpoint(model, args)
    # TODO
    #test_model(model, args, test_loader)
    #test_prob_and_vote(model, args, test_strokes)
    list_of_test_videos = get_videos_list(os.path.join(task_path, 'test'))
    test_videos_segmentation(model, args, list_of_test_videos)
    return 1

def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments defining stream information')
    parser.add_argument('--task','-t',default='dc',
                        choices=['dc', 'd', 'c'],
                        help='dc(detection and classification); d(detection); c(classification)')
    parser.add_argument('--model', '-m',default='V1',
                        help='choose model from model.py (e.g. V1, V2,...)')
    parser.add_argument('--stream_design','-sd',default='s',
                        choices=['s', 'srgb'],
                        help='s(skeleton); srgb(skeleton+rgb)')
    parser.add_argument('--epochs','-e', default=1000,
                        help='number of training epochs')
    #TODO: do we need model_load for c and d each? could be difficult
    parser.add_argument('--model_load','-ml', default=None,
                        help='load model from \'/working_folder/Models/<task_name>/model_load')
    parser.add_argument('--test_include','-ti',default='rgb',
                        choices=['rgb', 's', 'srgb', 'notest'],
                        help='rgb(include running test on rgb data); s; srgb; notest')
    parser.add_argument('--log_include','-li',default='nolog',
                        choices=['log', 'nolog'],
                        help='log(include writing log); nolog')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    '''
    Promt looks like this: python main_2.py -t <task> -m <model> -sd <stream_design> -e <epochs> -ti <test_include> -li <log_include>
    '''

    #args from terminal
    args = parse_args()

    # Chrono
    start_time = time.time()
    print()
    print('Start time: ', datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'))

    print('Running Modeling-Task with args: task:{}; model:{}; stream_design:{}; test_include:{}; log_include:{}'\
        .format(args.task, args.model, args.stream_design, args.test_include, args.log_include))

    print('Working GPU device:',torch.cuda.get_device_name(torch.cuda.current_device()))

    print()

    # MediaEval Task source folder
    source_folder = 'data'

    # Folder to save work
    working_folder = 'working_folder'

    #second stream design (besides rgb)
    if args.stream_design == 's':
        data_in = ['rgb', 's']
    elif args.stream_design == 'srgb':
        data_in = ['rgb', 'srgb']
    
    # Log file
    log_folder = os.path.join(working_folder, 'logs')
    os.makedirs(log_folder, exist_ok=True)
    if args.log_include == 'log':
        log = setup_logger('my_log', os.path.join(log_folder, '%s.log' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_{}-{}'.format(args.model, args.stream_design)))))
    elif args.log_include == 'nolog':
        log = None

    #TODO: cant we uncomment that part? Because everything is preprocessed already. Think it would be cleaner for final submission
    # Prepare work tree (respect levels for correct extraction of the frames)
    # create_working_tree(working_folder, source_folder, frame_width=320, log=log)
    print_and_log('Working tree created in %ds' % (time.time()-start_time), log=log)

    #Included data for test
    if args.test_include == 'rgb':
        test_include=get_videos_list(os.path.join(working_folder, 'rgb', 'detectionTask', 'test'))
    elif args.test_include == 's':
        test_include=get_videos_list(os.path.join(working_folder, 's', 'detectionTask', 'test'))
    elif args.test_include == 'srgb':
        test_include=get_videos_list(os.path.join(working_folder, 'srgb', 'detectionTask', 'test'))
    elif args.test_include == 'notest':
        test_include=None

    # Tasks
    if args.task=='dc':
        detection_task(working_folder, source_folder, data_in, args.epochs, args.model_load, log=log)
        classification_task(working_folder, data_in, args.epochs, args.model_load, test_strokes_segmentation=test_include, log=log)
    elif args.task=='d':
        detection_task(working_folder, source_folder, data_in, args.epochs, args.model_load, log=log)
    elif args.task=='c':
        classification_task(working_folder, data_in, args.epochs, args.model_load, test_strokes_segmentation=test_include, log=log)
    
    print_and_log('All Done in %ds' % (time.time()-start_time), log=log)