import argparse
import numpy as np
import os
import time

def arguments_work():
    a = np.zeros(shape=[4, 4, 3])
    print(a)

def parse_args():
    parser = argparse.ArgumentParser(description='Process some strings.')
    parser.add_argument('a',help='arg1')
    parser.add_argument('b',help='arg1')
    parser.add_argument('c',help='arg1')
    args = parser.parse_args()
    return args

def parse_args_running():
    args = parse_args()
    args_comb = 'Do',args.a,args.b,args.c,'?'
    print(args_comb)


def prefix_work(stream):
    if stream == 's':
        prefix = '_s'
    elif stream == 'srgb':
        prefix = '_srgb'
    print(prefix)

def iter_videos(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            print(os.path.join(subdir, file))

def iter_dir(rootdir):
    direction = os.listdir(rootdir)
    videos = []
    for subdir, dirs, files in os.walk(rootdir):
        print('subdir:', subdir)
        for file in files:
            if file.lower().endswith('.mp4'):
                videos.append(os.path.join(subdir,file))
                print('---------------',os.path.join(subdir,file))
        print('//////////')
        print('dir:', dir)
    print(videos)
    print(len(videos))

def print_root(rootdir):
    videos = os.walk(rootdir)
    print(videos)

def processing_time(sleeptime):
    print('Start sleeping for {}s...'.format(sleeptime))
    start_time = time.time()
    time.sleep(sleeptime)
    print('Actual processing time:{}'.format(round(((time.time()-start_time)/60),2)))

def os_join_test(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.lower().endswith('.mp4'):
                print(os.path.join(subdir,'{}'.format('s_')+file))

#arguments_work()
#parse_args_running()
#prefix_work('s')
#iter_videos('../data/classificationTask/validation/Serve Forehand Backspin')
#iter_dir('../data_lite')
#print_root('../data')
#processing_time(15)
os_join_test('../data_lite')