# VtacArm
Python code for vision-based tactile arm demonstration

Refer to paper: [VTacArm. A Vision-based Tactile Sensing Augmented Robotic Arm with Application to Human-robot Interaction](https://ieeexplore.ieee.org/document/9217019)

## Package version requirements
python 3.9.16

open3d 0.16.0

opencv-contrib-python 4.7.0

urx 0.11.0

numpy 1.23.5

## System tesing
1. Download branch [VtacArm](https://github.com/Guanlan-gkd/Ri-demo)

2. After plugging camera into computer, find camera index. Choose the upper number of /dev/video (0 in this case).
   
```
$ v4l2-ctl --list-devices 

Synaptics RMI4 Touch Sensor (rmi4:rmi4-00.fn54):
	/dev/v4l-touch0

Integrated Camera: Integrated C (usb-0000:00:14.0-8):
	/dev/video0
	/dev/video1

```

2. Repalce the camera index in [high_speed_camera_test.py](https://github.com/Guanlan-gkd/VtacArm/blob/main/high_speed_camera_test.py). Run the program. You should see the image from camera.
```
cap2 = cv2.VideoCapture(0) # change to camera index
```

