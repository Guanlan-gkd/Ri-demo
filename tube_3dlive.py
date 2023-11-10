import cv2
import numpy as np
from src.tracker import Tracker
import open3d as o3d
import time
import subprocess
print(o3d.__version__)
print(cv2.__version__)
# this command in terminal will list all the video devices

# v4l2-ctl --list-devices 
usb_list = subprocess.run(["v4l2-ctl","--list-devices"])
print("The exit code was: %d" % usb_list.returncode)
tracker = Tracker(adaptive=True,
                        cuda=False)  # cuda=True if using opencv cuda

cap2 = cv2.VideoCapture(2)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))


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

with open('pts.npy', 'rb') as f:
    tube_points = np.load(f)
    img_points = np.load(f)
    img2tube = np.load(f)
    mask = np.load(f)
    m = np.load(f)

m = mask * m
o3d_flag = True

print("init done")

if o3d_flag:
    tube = o3d.geometry.PointCloud()
    tube.points = o3d.utility.Vector3dVector(tube_points)
    tube.paint_uniform_color([0.5, 0.5, 0.5])

    contact = o3d.geometry.PointCloud()
    img_points_draw = []
    contact.points = o3d.utility.Vector3dVector(img_points_draw)
    contact.paint_uniform_color([1, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    vis.add_geometry(tube)
    vis.add_geometry(contact)

    # ctr.set_front([0, 0, 0])
    # ctr.set_lookat([0, 0, 0])
    # ctr.set_up([0, 0, 0])    
    ctr.rotate(0,-10000)


reset_flage = 1
t1 = time.time()
while True:    
    ret1, img = cap2.read()
    if ret1:
        # apply mask 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.flip(img, 1) # flip the image horizontally
        img = (img * mask).astype('uint8')
        # print(img.shape)
        # dst_roi = np.ascontiguousarray(img[90:-90, 120:-120])
        flow = tracker.track(img)
        # print(flow.shape)
        if reset_flage == 1 and time.time()- t1 > 1:
            tracker.reset()
            reset_flage = 0
            t1 = time.time()
            print("Tracker reset")
        mag = np.hypot(flow[:, :, 0], flow[:, :, 1])
        mag = mag*m*10 # magnify the magnitude, the factor 10 is for better visualization
        #print(mag.max(),mag.min())
        print(mag.max(),mag.min())
        mag = (mag > 0.2*max(mag.max(), 200)).astype('uint8')*255
        
        # M = cv2.moments(mag)            

        # # calculate x,y coordinate of center
        # if M["m00"] != 0:
        #     cX = int(M["m10"] / M["m00"])
        #     cY = int(M["m01"] / M["m00"])        
        # else:
        #     cX, cY = 312, 235
        # cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)

        if o3d_flag:
            img_points_contact = img_points * mag
            # cv2.imshow("img_point", img_points_contact)
            img_points_ind = np.nonzero(img_points_contact)
            img_points_draw = []
            # print(len(img_points_ind[0]))
            for i in range(len(img_points_ind[0])):
                img_points_draw.append(img2tube[img_points_ind[0][i], img_points_ind[1][i], :])
            contact.points = o3d.utility.Vector3dVector(img_points_draw)
            contact.paint_uniform_color([1, 0, 0])
            vis.update_geometry(contact)
            vis.poll_events()
            vis.update_renderer()
        
        
        # imag = cv2.applyColorMap(mag, cv2.COLORMAP_HOT)
        # print(imag.shape)
        arrows = put_optical_flow_arrows_on_image(
            cv2.cvtColor(img,cv2.COLOR_GRAY2BGR), flow[15:-15, 15:-15, :])
        # cv2.imshow('imag', imag)
        
        cv2.imshow('img', img)
        cv2.imshow('arrows', arrows)
        # cv2.imshow('mag', mag)
        
        k = cv2.waitKey(10)
        if k & 0xFF == ord('q'):
            break
            
        elif k & 0xFF == ord('r'):
            tracker.reset()
            print("Tracker reset")

vis.destroy_window()
cv2.destroyAllWindows()
        
    

  





