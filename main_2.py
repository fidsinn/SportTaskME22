import os
import time
import datetime
import torch

from utils import *
from init_data import create_working_tree


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

    # Tasks
    # detection_task(working_folder, source_folder, log=log)
    # classification_task(working_folder, log=log, test_strokes_segmentation=get_videos_list(os.path.join(working_folder, 'detectionTask', 'test')))
    
    print_and_log('All Done in %ds' % (time.time()-start_time), log=log)