'''
tello SDK Demo
'''

import time
from djitellopy import Tello
import cv2

class Demo:
    def __init__(self):
        # tello初始化
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()

    def camera(self):
        '''
        视频流
        '''
        # 电量
        batt = self.tello.query_battery()
        print('电量:'+str(batt))
        
       
        # cap = cv2.VideoCapture(0)

        while True:
            # ret,frame = cap.read()
            # 读取视频帧
            frame = self.tello.get_frame_read().frame
            # 镜像
            frame = cv2.flip(frame, 1)

            cv2.imshow('demo', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        cv2.destroyAllWindows()

    def flightControl(self):
        # 起飞
        self.tello.takeoff()
        '''
        向上向左飞，再降落
        '''
        self.tello.move_up(30)
        print('向上飞30CM')
        
        time.sleep(1)
        self.tello.move_left(30)
        print('向左飞30CM')
        time.sleep(3)
        self.tello.land()
        print('降落')

# 实例化
demo = Demo()
demo.flightControl()
# demo.camera()