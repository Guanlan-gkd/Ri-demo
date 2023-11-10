import cv2
import numpy as np
from src.tracker import Tracker
import math
import time
import urx
import logging
import subprocess
import os

def put_optical_flow_arrows_on_image(image, optical_flow, threshold=2.0):
    # Don't affect original image
    image = image.copy()

    scaled_flow = optical_flow * 2.0  # scale factor

    # Get start and end coordinates of the optical flow
    flow_start = np.stack(
        np.meshgrid(range(0, scaled_flow.shape[1], 30),
                    range(0, scaled_flow.shape[0], 30)), 2)
    flow_start[:,:,0] += 9
    flow_start[:,:,1] += 29
    flow_end = (scaled_flow[flow_start[:, :, 1], flow_start[:, :, 0], :] +
                flow_start).astype(np.int32)

    # Threshold values
    norm = np.linalg.norm(scaled_flow[flow_start[:, :, 1], flow_start[:, :,
                                                                      0], :],
                          axis=2)
    # print(norm.max(), norm.min())
    norm[norm < threshold] = 0
    # Draw all the nonzero values
    nz = np.nonzero(norm)
    norm = np.asarray((norm - norm.min())/ norm.max()* 255.0, dtype='uint8')
    # print(norm.max(), norm.min())
    color_image = cv2.applyColorMap(norm, cv2.COLORMAP_RAINBOW).astype('int')
    for i in range(len(nz[0])):
        y, x = nz[0][i], nz[1][i]
        cv2.arrowedLine(image,
                        pt1=tuple(flow_start[y, x]),
                        pt2=tuple(flow_end[y, x]),
                        color=(int(color_image[y, x, 0]), int(color_image[y, x, 1]),
                               int(color_image[y, x, 2])),
                        thickness=2,
                        tipLength=.3)
    return image


def divergence(flow):
    # divergence = F(x)/dx + F(y)/dy
    return np.sum(np.gradient(flow[:, :, 0],axis=1) + np.gradient(flow[:, :, 1],axis=0))

def curl(flow):
    # curl = F(y)/dx - F(x)/dy
    return np.sum(np.gradient(flow[:, :, 1],axis=1) - np.gradient(flow[:, :, 0],axis=0))

def move_(div, cur):
    # change treshhold to change sensitivity
    if div >= 8000:
        return 1
    elif div <= -8000:
        return 2
    elif cur >= 8000:
        return 3
    elif cur <= -8000:
        return 4
    else:
        return 0
    
with open('pts.npy', 'rb') as f:
    tube_points = np.load(f)
    img_points = np.load(f)
    img2tube = np.load(f)
    mask = np.load(f)
    m = np.load(f)
m = mask * m
move_type_list = ["up","down","left","right"]


if __name__ == "__main__":
    act_time = time.time()
    reset_flage = 1
    inc_flag = 0
    tracker = Tracker(adaptive=True,
                      cuda=False)  # cuda=True if using opencv cuda

    # this command in terminal will list all the video devices
    # v4l2-ctl --list-devices
    usb_list = subprocess.run(["v4l2-ctl","--list-devices"])
    print("The exit code was: %d" % usb_list.returncode)
    cap2 = cv2.VideoCapture(2)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))


    logging.basicConfig(level=logging.WARN)

    rob = urx.Robot("192.168.1.10")
    #rob = urx.Robot("localhost")
    rob.set_tcp((0,0,0,0,0,0))
    rob.set_payload(2, (0,0,0))
    
    l = 0.02
    v = 0.05
    a = 0.3
    pose = rob.getl()
    print("robot tcp is at: ", pose)
    j = rob.getj()
    print(j)
    j = [-3.1191824118243616, -1.5033019224749964, -1.9925110975848597, -1.2189400831805628, 1.5544813871383667, -1.5421856085406702]
    rob.movej(j, acc=a, vel=v)
    time.sleep(2)

    print("init done")
    input('Press enter to continue\n')

    while True:
        ret1, img = cap2.read()
        if ret1 == True:
            # apply mask
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (img * mask).astype('uint8')
            # print(img.shape)
            # dst_roi = np.ascontiguousarray(img[90:-90, 120:-120])
            flow = tracker.track(img)
            # print(flow.shape)
            if reset_flage == 1 and time.time()- act_time > 1:
                tracker.reset()
                reset_flage = 0
                act_time = time.time()
                print("Tracker reset")        


            arrows = put_optical_flow_arrows_on_image(
                cv2.cvtColor(img,cv2.COLOR_GRAY2BGR), flow[15:-15, 15:-15, :])
            cv2.imshow('arrows', arrows)
            cv2.imshow('img', img)

            k = cv2.waitKey(10)
            if k & 0xFF == ord('q'):
                break
                
            elif k & 0xFF == ord('r'):
                tracker.reset()
                print("Tracker reset")

            if reset_flage == 0:
                div = divergence(flow)
                cur = curl(flow)
                print(div, cur)

            if inc_flag== 0 and reset_flage != 1 and time.time() - act_time > 0.5:
                move_type = move_(div, cur)
                if move_type != 0:
                    inc_flag = 1
                    time.sleep(0.5)
                    # input("move type:" + move_type_list[move_type - 1])
                
                
                if move_type == 1:
                    rob.translate((0, 0, l), acc=a, vel=v)
                elif move_type == 2:
                    rob.translate((0, 0, -l), acc=a, vel=v)
                elif move_type == 3:
                    j = rob.getj()
                    j[5] += 0.17
                    rob.movej(j, acc=a, vel=0.2)
                elif move_type == 4:
                    j = rob.getj()
                    j[5] -= 0.17
                    rob.movej(j, acc=a, vel=0.2)
                act_time = time.time()
                
            # reset tracker after movement
            if inc_flag == 1 and time.time() - act_time > 0.5:
                tracker.reset()
                inc_flag = 0
                act_time = time.time()
                
            

            

            
        


    # motion.finish()
    # thread.join()
    cv2.destroyAllWindows()
    cap2.release()
    rob.close()
    print("bye~")


