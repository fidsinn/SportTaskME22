import os
import time
import datetime
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import json

from utils import *
from model import *
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
            self.move = LIST_OF_STROKES.index(move)

    def my_print(self, log=None):
        print_and_log('Video : %s\tbegin : %d\tEnd : %d\tClass : %s' % (self.video_path, self.begin, self.end, self.move), log=log)

'''
My_dataset class which uses My_stroke class to be used in the data loader
'''
class My_dataset(Dataset):
    def __init__(self, dataset_list, size_data, data_type, augmentation=False):
        self.dataset_list = dataset_list
        self.size_data = size_data
        self.augmentation = augmentation
        self.data_type = data_type

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        frames = get_data(self.dataset_list[idx].video_path, self.dataset_list[idx].begin, self.dataset_list[idx].end, self.size_data, self.augmentation)
        sample = {'data_type': torch.FloatTensor(frames), 'label' : self.dataset_list[idx].move, 'my_stroke' : {'video_path':self.dataset_list[idx].video_path, 'begin':self.dataset_list[idx].begin, 'end':self.dataset_list[idx].end}}
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

def get_data(data_path, begin, end, size_data, augmentation):
    frame_data = []
    if augmentation:
        angle, zoom, tx, ty, flip, begin = get_augmentation_parameters(begin, end, size_data)
        R_matrix = cv2.getRotationMatrix2D((size_data[0]//2-tx, size_data[1]//2-ty), angle, 1)
    else:
        angle, zoom, tx, ty, flip, begin = 0, 1, 0, 0, 0, max((begin+end-size_data[2])//2,0)
    
    max_frame_number = len(os.listdir(data_path))-1

    for frame_number in range(begin, begin + size_data[2]):
        if frame_number > max_frame_number:
            frame_number = max_frame_number
        try:
            frame = cv2.imread(os.path.join(data_path, '%08d.png' % frame_number))
            frame = cv2.resize(frame, (size_data[0], size_data[1])).astype(float) / 255
            if augmentation:
                frame = apply_augmentation(frame, zoom, R_matrix, flip)
        except:
            raise ValueError('Problem with %s begin %d size %s' % (os.path.join(data_path, '%08d.png' % frame_number), begin, str(size_data)))
        frame_data.append(cv2.split(frame))
    # To match size_data variable, transposition is needed. From (T,C,H,W) to (C,W,H,T).
    frame_data = np.transpose(frame_data, (1, 3, 2, 0))
    return frame_data




'''
Build dataloader from list of strokes
'''
def get_data_loaders(train_strokes, validation_strokes, test_strokes, size_data, batch_size, workers, data_type):
    # Build Dataset
    train_set = My_dataset(train_strokes, size_data, data_type)
    validation_set = My_dataset(validation_strokes, size_data, data_type)
    test_set = My_dataset(test_strokes, size_data, data_type)

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
    model = CCNAttentionNet_TwoStream(args.size_data.copy(), output_size)
    print_and_log('Model %s created' % (model.__class__.__name__), log=args.log)
    ## Use GPU
    if args.cuda:
        model.cuda()
    return model

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
        rgb, label = batch['rgb'], batch['label']

        rgb = Variable(rgb.type(args.dtype))
        label = Variable(label.type(args.dtype).long())

        optimizer.zero_grad()
        output = model(rgb)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        aLoss += loss.item()
        Acc += output.data.max(1)[1].eq(label.data).cpu().sum().numpy()
        progress_bar((batch_idx + 1) * args.batch_size, N, '%d Training - Epoch : %d - Batch Loss = %.5g' % (pid, epoch, loss.item()))

    aLoss /= N
    progress_bar(N, N, 'Train - Epoch %d - Loss = %.5g - Accuracy = %.3g (%d/%d) - Time = %ds' % (epoch, aLoss, Acc/N, Acc, N, time.time() - begin_time), 1, log=args.log)
    return aLoss, Acc/N


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
    return train_strokes, validation_strokes, test_strokes

def classification_task(working_folder, data_in = ['rgb', 's'], log=None, test_strokes_segmentation=None):
    '''
    Main of the classification task
    Perform also on the detection task when the videos for segmentation are provided
    '''
    print_and_log('\nClassification Task', log=log)
    # Initialization
    reset_training(1)
    task_name = 'classificationTask-' + data_in[0] + '-' + data_in[1] 
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
    args = My_variables(working_folder, task_name)
    
    # Architecture with the output of the lenght of possible classes - (Unknown not counted)
    # make two identical models
    model = make_architecture(args, len(LIST_OF_STROKES))

    # Loaders
    train_loader_list, validation_loader_list, test_loader_list = [], [], []
    for i in enumerate(data_in):
        train_loader, validation_loader, test_loader = get_data_loaders(train_strokes_list[i], validation_strokes_list[i], test_strokes_list[i], args.size_data, args.batch_size, args.workers)
        train_loader_list.append(train_loader)
        validation_loader_list.append(validation_loader)
        test_loader_list.append(test_loader_list)

    # Training process
    if args.train_model:
        train_model(model, args, train_loader, validation_loader)
    
    # Test process
    load_checkpoint(model, args)
    test_model(model, args, test_loader, list_of_strokes=LIST_OF_STROKES)
    test_prob_and_vote(model, args, test_strokes, list_of_strokes=LIST_OF_STROKES)
    if test_strokes_segmentation is not None:
        test_videos_segmentation(model, args, test_strokes_segmentation, sum_stroke_scores=True)
    return 1



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