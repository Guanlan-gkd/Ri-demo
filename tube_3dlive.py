import cv2
import numpy as np
import src.utils as utils
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


# img = cv2.imread("./tube_img/1.jpg", cv2.IMREAD_GRAYSCALE)
# apply mask
# img = cv2.circle(img, (312, 235), 40, (0, 0, 0), -1)
# cv2.imshow('img', img)
# cv2.waitKey()
# print(img.shape)

def generate_mask():
    '''
    this funciton generate mask to cover the center of the image, which is the tube top, so that the light change
    in that area will not affect the dis optical flow tracking.
    '''
    center = (325, 258)
    r1 = 40
    r2 = 320
    x = np.mgrid[0:640]
    y = np.mgrid[0:480]
    xv, yv = np.meshgrid(x, y)
    mask = np.ones_like(xv)
    print(mask.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (xv[i,j] - center[0])**2 + (yv[i,j] - center[1])**2 < r1**2:
                mask[i,j] = 0
            elif (xv[i,j] - center[0])**2 + (yv[i,j] - center[1])**2 > r2**2:
                mask[i,j] = 0
    return mask

# def tube_points():
#     Height = 200
#     Radius = 37
#     f = 250


#     center = (312, 235)

#     x = np.mgrid[0:640]
#     y = np.mgrid[0:480]
#     xv, yv = np.meshgrid(x, y)
#     xv = xv.reshape((-1,1))
#     yv = yv.reshape((-1,1))
#     # print(xv.shape, yv.shape)
#     r = np.sqrt((xv - center[0])**2 + (yv - center[1])**2)
#     xt = Radius * (xv - center[0]) / r
#     yt = Radius * (yv - center[1]) / r
#     zt = Radius * f / r
#     zt = zt - np.min(zt)
#     xyzt = np.ndarray.tolist(np.dstack((xt,yt,zt)).reshape((-1,3)))
#     print(len(xyzt), len(xyzt[0]))

#     for i in range(len(xyzt)-1,-1,-1):
#         if (xv[i] - center[0])**2 + (yv[i] - center[1])**2 < 40**2:
#             xyzt.pop(i)

#     # np.savetxt("xyzt.csv", xyzt, delimiter=",")
#     return xyzt



def magnify_field():
    '''
    this function generate a filed which later will be used to magnify the optical flow magnitude to compensate 
    the small magnitude in the tube center due to projection view. The closer to center on image means the further from the camera,
    therefore the larger the magnitude.
    '''
    center = (325, 258)
    x = np.mgrid[0:640]
    y = np.mgrid[0:480]
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt((xv - center[0])**2 + (yv - center[1])**2)
    m = np.nan_to_num(np.ones_like(r) / np.power(r, 1) * np.power(400, 1))
    
    return m
    # print(xv.shape, yv.shape)

def get_point(density_r = 500, density_h = 40):
    '''
    this function generate interpolated 3d points it returns 2 arrays: tube_points and img_points
    img_points is the projection of tube_points on the image plane
    density_r and _h  controls the number of points generated (_r * _h)
    '''
    Height = 150
    Radius = 37
    # generate tube_points
    h = np.mgrid[0:Height:density_h*1j]
    rad = np.mgrid[0:2*np.pi:density_r*1j]
    x = np.cos(rad) * Radius
    y = np.sin(rad) * Radius 
    x1 = np.repeat(x, h.shape[0]).reshape((-1,1))
    y1 = np.repeat(y, h.shape[0]).reshape((-1,1))
    h1 = np.tile(h, (x.shape[0],1)).reshape((-1,1))
    # print("x1", x1.shape)
    # print("y1", y1.shape)
    # print("h1", h1.shape)
    tube_points = np.stack((x1, y1, h1), axis=1).reshape((-1,3)) # tube_points is the 3d points
    print("tube_points", tube_points.shape)
    
    f = 200
    center = (325, 258)
    img_rad = np.repeat(rad, h.shape[0]).reshape((-1,1))
    img_radius = abs((f * Radius / (h1+26))).astype('int')
    # print("img radius", img_radius.shape)
    img_x = np.cos(img_rad) * img_radius + center[0]
    img_y = np.sin(img_rad) * img_radius + center[1]
    # print(img_x.shape, img_y.shape)
    img_points = np.zeros((480,640))
    img2tube = np.zeros((480,640,3))
    for i in range(img_x.shape[0]):
        if 0 < img_y[i] < 480 and 0 < img_x[i] < 640:
            img_points[int(img_y[i]), int(img_x[i])] = 1
            img2tube[int(img_y[i]), int(img_x[i]), :] = tube_points[i,:]
    # cv2.imshow("img_point1", img_points)
    
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # cv2.imwrite("img_points.png", (img_points*255).astype('uint8'))

    return tube_points, img_points, img2tube


# xyz_tube = tube_points()
# xyz_len = xyz_tube.shape[0]
mask = generate_mask()
m = magnify_field()
m = mask * m
# np.savetxt("mask.csv", mask, delimiter=",")
# np.savetxt("m.csv", m, delimiter=",")
tube_points, img_points, img2tube = get_point()
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
        
    

  





