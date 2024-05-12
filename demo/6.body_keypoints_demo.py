"""

mediapipe关键点演示

"""

import cv2
import numpy as np
import time

# mediapipe
from KEYPOINTS import Mpkeypoints

# 实例化关键点提取
mp_keypoints = Mpkeypoints()


class Ai_tello:
    def __init__(self):
      pass

    def processKeypoints(self,frame,p_list):
        '''
        关键点画线
        识别出对应动作
        '''
        # 关键点连接关系
        connectoions = [
            [0,1],
            [1,2],
            [2,3],
            [3,7],
            [0,4],
            [4,5],
            [5,6],
            [6,8],
            [9,10],
            [11,12],
            [12,14],
            [14,16],
            [16,18],
            [16,20],
            [16,22],
            [11,13],
            [13,15],
            [15,17],
            [15,19],
            [15,21],
            [12,24],
            [24,26],
            [26,28],
            [28,30],
            [28,32],
            [11,23],
            [23,25],
            [25,27],
            [27,29],
            [27,31],
            [23,24],
        ]
        # 画线        
        for conn in connectoions:
            # p_list[conn[0]] 连线的一个点
            # p_list[conn[1]] 连线的一个点
            # [100,233]
            cv2.line(frame,p_list[conn[0]],p_list[conn[1]],(0,255,0),2)

        # 画点
        for p_i,(x,y) in enumerate(p_list):
            # 鼻尖中心点
            cv2.circle(frame,(x,y),5,(255,0,255),-1)
        


        return None


    def cameraProcess(self):
        '''
        视频流处理：动作识别、绘制等
        '''
        
        cap = cv2.VideoCapture(0)
        # 动作
    

        while True:

            # 读取视频帧
            ret,frame = cap.read()

            frame = cv2.flip(frame, 1)

            # 计算视频高度和宽度
            frame_h, frame_w = frame.shape[:2]

            # 转为RGB
            img_cvt = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # 关键点检测
            pose_landmarks = mp_keypoints.getFramePose(img_cvt)

            # 判断画面中是否有关键点：
            if  pose_landmarks:

                # 解析关键点，并且恢复为像素值 [[100,223],[122,344]]
                p_list = [[int(landmark.x * frame_w),int(landmark.y * frame_h )] for landmark in pose_landmarks.landmark]
                # 转为numpy，才能广播计算
                p_list = np.asarray(p_list)
                # print(p_list)

                # 原图上绘制关节点及、连线
                self.processKeypoints(frame,p_list)


            cv2.imshow('demo', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        cv2.destroyAllWindows()



if __name__ == '__main__':

    # 实例化
    ai_tello = Ai_tello()
    ai_tello.cameraProcess()

