import os
from PIL import Image
import pickle

def resize_videos(load_path, save_path, new_width=224, new_height=224):
    """
    load_path: path of videos to be resized
    save_path: path of resized videos to be saved
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        assert False, 'Cannot create directory. Directory already exists.'

    with open('/project/DALY/all_videos.pkl', 'rb') as f:
        all_videos = pickle.load(f)

    for video_idx, video in enumerate(all_videos):
        print('Video', video_idx, '/', len(all_videos))
        os.mkdir(os.path.join(save_path, video))
        video_path = os.path.join(load_path, video)
        video_frames = os.listdir(video_path)
        for frame in video_frames:
            img = Image.open(os.path.join(video_path, frame))
            assert (new_width <= img.width) and (new_height <= img.height)
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
            img.save(os.path.join(save_path, video, frame))

def resize_videos2(load_path):
    
    with open('/project/DALY/annot_data_keyframes2.pkl', 'rb') as f:
        annot_data = pickle.load(f)
    
    sizes = [224, 256, 272, 300, 316]
    for size in sizes:
        save_path = '/project/DALY/daly_frames_resized_' + str(size) + '_' + str(size)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        else:
            assert False, 'Cannot create directory. Directory already exists.'
        
    video_idx = 0
    for split in ['training', 'validation', 'test']:
        all_videos = list(annot_data[split].keys())
        for video in all_videos:

            print('Video', video_idx + 1, '/', 510)
            for size in  sizes:
                save_path = '/project/DALY/daly_frames_resized_' + str(size) + '_' + str(size)
                os.mkdir(os.path.join(save_path, video))
            video_path = os.path.join(load_path, video)

            video_id = video.split('.')[0]
            instances = list(annot_data[split][video]['action_instances'].keys())
            video_frames = os.listdir(video_path)
            for instance in instances:
                start, end = annot_data[split][video]['action_instances'][instance]['tbound']
                start = start - 50
                end = end + 50
                frames = ['frame' + str(frame).zfill(6) + '.jpg' for frame in range(start, end + 1)]
                for frame in frames:
                    if frame not in video_frames:
                        continue
                        
                    img = Image.open(os.path.join(video_path, frame))
                    #assert (size <= img.width) and (size <= img.height)
                    
                    size = 224
                    save_path = '/project/DALY/daly_frames_resized_' + str(size) + '_' + str(size)
                    resized_img = img.resize((size, size), Image.ANTIALIAS)
                    resized_img.save(os.path.join(save_path, video, frame))
                    
                    size = 256
                    save_path = '/project/DALY/daly_frames_resized_' + str(size) + '_' + str(size)
                    resized_img = img.resize((size, size), Image.ANTIALIAS)
                    resized_img.save(os.path.join(save_path, video, frame))
                    
                    size = 272
                    save_path = '/project/DALY/daly_frames_resized_' + str(size) + '_' + str(size)
                    resized_img = img.resize((size, size), Image.ANTIALIAS)
                    resized_img.save(os.path.join(save_path, video, frame))
                    
                    size = 300
                    save_path = '/project/DALY/daly_frames_resized_' + str(size) + '_' + str(size)
                    resized_img = img.resize((size, size), Image.ANTIALIAS)
                    resized_img.save(os.path.join(save_path, video, frame))
                    
                    size = 316
                    save_path = '/project/DALY/daly_frames_resized_' + str(size) + '_' + str(size)
                    resized_img = img.resize((size, size), Image.ANTIALIAS)
                    resized_img.save(os.path.join(save_path, video, frame))
                    
            video_idx += 1

            

def rename_videos(load_path):
    os.chdir(load_path)
    """
     Renames the video file from a full name to video id.
     Warning: The function will rename the file in-place.
    """
    videos = os.listdir('.')
    for video in videos:
        video_id = video[-15:]
        os.rename(video, video_id)

def check_nbframes_ffmpeg(path):
    """ 
    Checks whether the number of frames extracted from each video using ffmpeg match those of annotations.
    path: path of extracted frames
    """
    
    with open('/project/DALY/daly1.1.0.pkl', 'rb') as f:
        annot = pickle.load(f, encoding='latin1')
    daly_cache = os.listdir('/project/DALY/daly_cache/daly_videos')
    
    eq_frames = []
    uneq_videos = []
    videos = os.listdir(path)
    for video in videos:
        #frame1 = os.listdir(os.path.join('/project/DALY/extracted_frames/', video))[0]
        #frame2 = os.listdir(os.path.join('/project/DALY/daly_cache/daly_images', video))[0]
        n_frames_ffmpeg = len(os.listdir(os.path.join(path, video)))
        n_frames_annot = annot['metadata'][video]['nbframes_ffmpeg']
        #w1, h1, _ = np.array(Image.open(os.path.join('/project/DALY/extracted_frames/', video, frame1))).shape
        #w2, h2, _ = np.array(Image.open(os.path.join('/project/DALY/daly_cache/daly_images', video, frame2))).shape
        #print(w2, h2)
        #if (w1 == w2) & (h1 == h2):
        #if (n_frames_ffmpeg != n_frames_annot):
        #    print(n_frames_ffmpeg - n_frames_annot)
        #if video == 'sjgkGNQyOWg.mp4':
        #    print(n_frames_ffmpeg - n_frames_annot)
        if n_frames_ffmpeg != n_frames_annot:
            print(abs(n_frames_ffmpeg) - n_frames_annot)
            uneq_videos.append(video)
            #print(n_frames_ffmpeg - n_frames_annot, video)
    return(uneq_videos)        

def get_frames_size():
    """ 
    Outputs a dictionary containing the frames' size (width, height) of each video.
    """
    frames_size = {} 
    with open('/project/DALY/all_videos.pkl', 'rb') as f:
        all_videos = pickle.load(f)
    cache_dir = '/project/DALY/daly_cache/daly_images/'
    for video in all_videos:
        keyframe = os.listdir(os.path.join(cache_dir, video))[0]
        keyframe_array = np.array(Image.open(os.path.join(cache_dir, video, keyframe)))
        height = keyframe_array.shape[0]
        width = keyframe_array.shape[1]
        frames_size[video] = (width, height)
    return(frames_size)

def get_frames_size_pkl(path='/project/DALY/'):
    """ 
    Loads a dictionary containing the frames' size (width, height) of each video.
    Same result as calling get_frames_size(), but faster.
    """
    with open(os.path.join(path, 'frames_size.pkl'), 'rb') as f:
        frames_size =  pickle.load(f)
    return frames_size    
