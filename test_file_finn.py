import argparse
import numpy as np
import os
import time
import datetime
import shutil

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

def weird_for_loop():
    test_list_two = [1, 2, 3]
    test_list = ['a','b','c']
    test_list_three = [
        test_list_two.append(i)
        for i in range(test_list_two)
    ]
    
def concatting():
    string1 = '1234.mp4'
    string2 = 's_' + string1
    return string2

def remove_wrong_paths():
    rootdir = 'GIT/SportTaskME22/working_folder/srgb'
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            print(d)

def listdirs(rootdir):
    for it in os.scandir(rootdir):
        if it.is_dir():
            if 's_' in it.path or 'srgb_' in it.path:
                print(it.path)
                listdirs(it)

def list_directs(directory):
    #list_directs = [folder[0] for folder in os.walk(directory) if 's_' in folder or 'srgb_' in folder]
    list_directs = [f.path for f in os.scandir(directory) if f.is_dir()]
    for directory in list(list_directs):
        list_directs.extend(list_directs(directory))
    return list_directs

def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    matching = [s for s in subfolders if 's_' in s or 'srgb_' in s]
    for m in matching:
        shutil.rmtree(m)
    return matching

def timetime():
    print(datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
    print(datetime.date.today())

#arguments_work()
#parse_args_running()
#prefix_work('s')
#iter_videos('../data/classificationTask/validation/Serve Forehand Backspin')
#iter_dir('../data_lite')
#print_root('../data')
#processing_time(15)
#os_join_test('data_lite')
#weird_for_loop()
#print(concatting())
#remove_wrong_paths()
rootdir = 'GIT/SportTaskME22/working_folder/s' #dann nochmal für /rgb ausführen
#listdirs(rootdir)
#list_directs(rootdir)
#print(fast_scandir(rootdir))
timetime()