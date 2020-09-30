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
