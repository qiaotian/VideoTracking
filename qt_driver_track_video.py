# @Author: Tian Qiao <qiaotian>
# @Date:   2016-06-13T19:29:04+08:00
# @Email:  qiaotian@me.com
# @Last modified by:   qiaotian
# @Last modified time: 2016-06-14T13:20:12+08:00
# @License: Free License



import cv2
import argparse
import numpy as np

drawing = False #
roi = [0,0,0,0] # x, y, w, h
img = np.zeros((640, 480, 3), np.uint8)
heart = [0,0] # gravity center(row, col)

# mouse callback function
def select_roi(event, x, y, flag, param):
    global roi,drawing,img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi[0] = x
        roi[1] = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True and x>roi[0] and y>roi[1]:
            roi[2] = x-roi[0]
            roi[3] = y-roi[1]
            cv2.rectangle(img, (roi[0], roi[1]), (x, y), (0,255,0), -1)

            heart[0] = roi[0]
            heart[1] = roi[1]
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False



def highlight(image, threshold, roi):
    assert image.ndim == 3 # 3 channel image
    sum_row = 0
    sum_col = 0
    count = 0
    for i in range(roi[1], roi[1]+roi[3]):
        for j in range(roi[0], roi[0]+roi[2]):
            if image[i][j][0] > threshold or image[i][j][1] > threshold or image[i][j][2] > threshold:
                image[i][j] = [255,255,255] # white
                sum_row = sum_row + i
                sum_col = sum_col + j
                count=count+1
            else:
                image[i][j] = 0 # black
    heart[0] = sum_row//count
    heart[1] = sum_col//count

    return image

def main(arv=None):
    global img, roi, heart
    parser = argparse.ArgumentParser(
        description='Crop the image and store the croped image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input_filename', default='../ExperimentData/usdata/croped_MOVIE-0012.mp4', help='')
    parser.add_argument('--output_filename', default='./res/hightlight_center.avi', help='')
    parser.add_argument('--log', default='./res/hightlight_center_log.txt', help='')
    parser.add_argument('--threshold', type = np.float32, default=150, help='')
    args = parser.parse_args()

    # select the roi
    videoCapture = cv2.VideoCapture(args.input_filename)
    if(videoCapture.isOpened()):
        ret, img = videoCapture.read()
        if ret==True:
            #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            cv2.namedWindow('select_roi')
            cv2.setMouseCallback('select_roi', select_roi)
        else:
            cv2.destroyAllWindows()
            return
    while(1):
        cv2.imshow('select_roi', img)
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            cv2.destroyAllWindows()
            break


    # play the tracking video with roi
    f = open(args.log, "w+") # overwrite

    videoCapture = cv2.VideoCapture(args.input_filename)
    fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.cv.FOURCC('8', 'B', 'P', 'S') # works
    videoWriter = cv2.VideoWriter(args.output_filename, fourcc, fps, size, True)
    while(videoCapture.isOpened()):
        ret, frame = videoCapture.read()
        if ret == True:
            #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            disp = highlight(frame, args.threshold, roi)
            cv2.circle(disp, (heart[1], heart[0]), 10, (0,255,255), -1) # draw the gravity center
            cv2.imshow('highlight', disp)
            videoWriter.write(disp)
            f.write('%d %d\n' % (heart[1], heart[0]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    f.close()
    return


if __name__ == '__main__':
    main()
