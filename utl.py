# @Author: Tian Qiao
# @Date:   2016-06-05T10:21:02+08:00
# @Email:  qiaotian@me.com
# @Last modified by:   root
# @Last modified time: 2016-06-05T12:35:36+08:00
# @License: DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE



import cv2
import numpy as np
import argparse
import time

def crop(image, x, y, w, h):
    if image.ndim != 3:
        print('image dim invalid, crop failed')
        return
    if x<0 or y<0 or x+w>=image.shape[1] or x+w>=image.shape[0]:
        print('index beyond the boundry')
    return image[x:x+w-1, y:y+h-1, :]

def main(arv=None):
    parser = argparse.ArgumentParser(
        description='Crop the image and store the croped image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--image')
    parser.add_argument('--x', type=int)
    parser.add_argument('--y', type=int)
    parser.add_argument('--w', type=int, help='width')
    parser.add_argument('--h', type=int, help='height')
    args = parser.parse_args()

    image = cv2.imread(args.image)
    croped = crop(image, args.x, args.y, args.w, args.h)
    if croped.shape[0]!=0:
        cv2.imwrite('./out/%s.jpg' % time.time(), croped)
    else:
        print('croped image is None')

if __name__ == '__main__':
    main()
