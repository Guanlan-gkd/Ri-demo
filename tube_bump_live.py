import cv2
import numpy as np
from src.tracker import Tracker
import math
import time
import urx
import logging
import subprocess

def put_optical_flow_arrows_on_image(image, optical_flow, threshold=4.0):
    # Don't affect original image
    image = image.copy()

    scaled_flow = optical_flow * 2.0  # scale factor

    # Get start and end coordinates of the optical flow
    flow_start = np.stack(
        np.meshgrid(range(0, scaled_flow.shape[1], 30),
                    range(0, scaled_flow.shape[0], 30)), 2)
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
                        thickness=1,
                        tipLength=.3)
    return image

def divergence(flow):
    # divergence = F(x)/dx + F(y)/dy
    return np.sum(np.gradient(flow[:, :, 0],axis=1) + np.gradient(flow[:, :, 1],axis=0))

def curl(flow):
    # curl = F(y)/dx - F(x)/dy
    return np.sum(np.gradient(flow[:, :, 1],axis=1) - np.gradient(flow[:, :, 0],axis=0))
    
def bump_move(cX,cY,move_dis = 0.05):
    cX = cX - 325
    cY = cY - 258
    # move distance in meter
    disX = move_dis / math.sqrt(cX**2 + cY**2) * cX
    disY = move_dis / math.sqrt(cX**2 + cY**2) * cY
    disXtran = (disX - disY) * 0.707
    disYtran = (disX + disY) * 0.707
    return round(disXtran,4), round(disYtran,4)

with open('pts.npy', 'rb') as f:
    tube_points = np.load(f)
    img_points = np.load(f)
    img2tube = np.load(f)
    mask = np.load(f)
    m = np.load(f)
m = mask * m

if __name__ == "__main__":

    reset_flage = 1
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
    time.sleep(3)
    print("init done")
    input('Press enter to continue\n')
    t1 = time.time()
    act_time = time.time()
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
            if reset_flage == 1 and time.time()- t1 > 1.:
                tracker.reset()
                reset_flage = 0
                t1 = time.time()
                print("Tracker reset")
            mag = np.hypot(flow[:, :, 0], flow[:, :, 1])
            mag = mag*m*10 # magnify the magnitude, the factor 10 is for better visualization
            #print(mag.max(),mag.min())
            print(mag.max(),mag.min())
            mag = (mag > 0.2*max(mag.max(), 200)).astype('uint8')*255
            M = cv2.moments(mag)            

            # calculate x,y coordinate of center
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])        
            else:
                cX, cY = 325, 258
            cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)

            cv2.imshow('img', img)
            k = cv2.waitKey(10)
            if k & 0xFF == ord('q'):
                break
                
            elif k & 0xFF == ord('r'):
                tracker.reset()
                print("Tracker reset")

            if (cX-312)**2 + (cY-235)**2 > 40**2:
                if time.time() - act_time > 0.5:
                    disy, disx = bump_move(cX, cY, move_dis=0.05)
                    # input("dX = {}, dY = {}".format(disx*1000, disy*1000))
                    rob.translate((disy, -disx, 0), acc=a, vel=v)
                    act_time = time.time()
                else:
                    tracker.reset()
            
            # imag = cv2.applyColorMap(mag, cv2.COLORMAP_HOT)
            # print(imag.shape)
            # arrows = put_optical_flow_arrows_on_image(
                # cv2.cvtColor(img,cv2.COLOR_GRAY2BGR), flow[15:-15, 15:-15, :])
            # cv2.imshow('imag', imag)
            
            # cv2.imshow('boundary', contact_boundary)
            # cv2.imshow('arrows', arrows)
            # cv2.imshow('mag', mag)
            
        


    # motion.finish()
    # thread.join()
    cv2.destroyAllWindows()
    cap2.release()
    rob.close()
    print("bye~")


