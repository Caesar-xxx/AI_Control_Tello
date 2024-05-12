'''
yolo 动作训练图片采集程序
'''
import cv2
import numpy as np
import time
# tello
from djitellopy import Tello

class Collect:
    def __init__(self):
        # tello初始化
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()


    def main(self,label_id=1,counts=3):

        index = 1
        last_col_time =  None

        while True:
            # 读取视频帧
            frame = self.tello.get_frame_read().frame
            
            if last_col_time is None:
                last_col_time =  time.time()

            # 翻转
            frame = cv2.flip(frame, 1)

            if index < counts+1:
                now = time.time()
                if now - last_col_time > 1:
                    # 每隔1秒采集一张: 1_1.jpg, 1_2.jpg
                    filename = './trainingImgs/{}_{}.jpg'.format(label_id,index)
                    cv2.imwrite(filename,frame)
                    
                    last_col_time = now

                    # 获取电量
                    battery = self.tello.get_battery()

                    print('成功采集：{}，飞机电量：{}'.format(filename,battery))

                    index +=1


            else:
                
                break
                
            
            # 显示
            cv2.imshow('demo', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        cv2.destroyAllWindows()


        
# 实例化
collect = Collect()
# 给你准备5s
time.sleep(5)
# 标签为1，采集15张
collect.main(3,30)