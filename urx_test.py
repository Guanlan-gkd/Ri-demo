import urx
import logging
import math3d as m3d
import numpy as np
# print(urx.__version__)
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)

    rob = urx.Robot("192.168.1.102")
    #rob = urx.Robot("localhost")
    rob.set_tcp((0,0,0,0,0,0))
    rob.set_payload(2, (0,0,0))

    l = 0.02
    v = 0.05
    a = 0.3
    j = rob.getj()
    print(j)
    # 0.17 in j[5] is 10 degree
    pose = rob.getl()
    print("robot tcp is at: ", pose)
    print(j)
    print("absolute move in base coordinate ")
    # pose[2] += l
    # rob.movel(pose, acc=a, vel=v)
    # j[5] += 0.17
    # rob.movej(j, acc=a, vel=v)
    rob.close()