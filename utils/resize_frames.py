import sys
import video_utils

load_path = sys.argv[1]
save_path = sys.argv[2]
width = int(sys.argv[3])
height = int(sys.argv[4])

size = (width, height)

video_utils.resize_frames(load_path, save_path, size)
