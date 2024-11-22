import os.path as osp
import os
import subprocess
import cv2

def video_to_images(vid_file, img_folder=None, fps=None, max_frames=None, return_info=False):
    if img_folder is None:
        img_folder = osp.join('/tmp', osp.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2']
    if fps is not None:
        command.extend(['-vf', f'fps={fps}'])
    if max_frames is not None:
        command.extend(['-vframes', str(max_frames)])
    command.extend(['-v', 'error',
                   f'{img_folder}/%05d.png'])
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '00001.png')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder
    
def images_to_video(img_folder, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-y', '-threads', '16', '-i', f'{img_folder}/%06d.png', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--video_name", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--fps", type=int, default=100)
    args = parser.parse_args()

    video_path = osp.join(args.video_folder, args.video_name)
    output_path = osp.join(args.video_folder, args.output_folder)
    video_to_images(video_path, output_path, fps=args.fps)
