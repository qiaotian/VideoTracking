# @Author: Tian Qiao
# @Date:   2016-06-05T10:21:02+08:00
# @Email:  qiaotian@me.com
# @Last modified by:   qiaotian
# @Last modified time: 2016-06-13T18:25:25+08:00
# @License: DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE



import cv2
import numpy as np
import argparse
import time
import os

def roi(image, x, y, w, h):
    if image.ndim != 3:
        print('image dim invalid, crop failed')
        return
    if x<0 or y<0 or x+w>=image.shape[1] or x+w>=image.shape[0]:
        print('index beyond the boundry')
    return image[x:x+w-1, y:y+h-1, :]

def crop_image(input_filename, output_filename, x, y, w, h):
    # TODO:
    return

def crop_image_batch(input_dir, output_dir, x, y, w, h):
    return

def crop_video(input_filename, output_filename, x, y, w, h):
    videoCapture = cv2.VideoCapture(input_filename)
    fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
    origin_size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
                   int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    # check the validation of params x,, y, w, h
    assert(x+w < origin_size[0] and y+h < origin_size[1])

    size = (w,h)
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')
    #fourcc = cv2.cv.FOURCC('M', 'P', '4', 'V')
    fourcc = cv2.cv.FOURCC('8', 'B', 'P', 'S')
    videoWriter  = cv2.VideoWriter(output_filename, fourcc, fps, size, True)

    while(videoCapture.isOpened()):
        ret, frame = videoCapture.read()
        if ret==True:
            videoWriter.write(frame[y:y+h, x:x+w])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    videoCapture.release()
    videoWriter.release()

def crop_video_batch(input_dir, output_dir, x, y, w, h):
    list_dirs = os.walk(input_dir)
    for root, dirs, files in list_dirs:
        for f in files:
            if f.endswith('mp4'):
                if input_dir.endswith('/'):
                    crop_video(input_dir+f, output_dir+'croped_'+f, x, y, w, h)
                else:
                    crop_video(input_dir+'/'+f, output_dir+'/croped_'+f, x, y, w, h)
                print(f, ' has been croped')
    return

def main(arv=None):
    parser = argparse.ArgumentParser(
        description='Crop the image and store the croped image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    """
    parser.add_argument('--image')
    parser.add_argument('--x', type=int)
    parser.add_argument('--y', type=int)
    parser.add_argument('--w', type=int, help='width')
    parser.add_argument('--h', type=int, help='height')
    args = parser.parse_args()

    image = cv2.imread(args.image)
    croped = roi(image, args.x, args.y, args.w, args.h)
    if croped.shape[0]!=0:
        cv2.imwrite('./out/%s.jpg' % time.time(), croped)
    else:
        print('croped image is None')
    """
    """
    parser.add_argument('-i', '--input_filename', default='../ExperimentData/usdata/MOVIE-0001.mp4')
    parser.add_argument('-o', '--output_filename', default='../ExperimentData/usdata/croped_MOVIE-0001.avi')
    parser.add_argument('-x', '--origin_x', type = int, default=300)
    parser.add_argument('-y', '--origin_y', type = int, default=300)
    parser.add_argument('-w', '--width', type = int, default=50)
    parser.add_argument('-h', '--height', type = int, default=50)

    args = parser.parse_args()
    crop_video(args.input_dir, args.output_dir, args.x, args.y, args.w, args.h)
    """
    parser.add_argument('--input_dir', default='../ExperimentData/usdata/', help='')
    parser.add_argument('--output_dir', default='../ExperimentData/usdata/', help='')
    parser.add_argument('--origin_x', type = int, default=30, help='')
    parser.add_argument('--origin_y', type = int, default=60, help='')
    parser.add_argument('--width', type = int, default=480, help='')
    parser.add_argument('--height', type = int, default=400, help='')

    args = parser.parse_args()
    crop_video_batch(args.input_dir, args.output_dir, args.origin_x, args.origin_y, args.width, args.height)

if __name__ == '__main__':
    main()
