"""
测试训练号的YOLO模型
"""

import cv2
import numpy as np
from djitellopy import Tello

import torch

class yolo_test:
    def __init__(self):

        # tello初始化
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()

     
        # 加载 yolov5模型
        self.model = torch.hub.load('./yolov5', 'custom', path='./weights/pose.pt',source='local') 
        # 置信度阈值
        self.model.conf = 0.7

    def demo(self):
       
        while True:

            # 读取视频帧
            frame = self.tello.get_frame_read().frame
            # 镜像
            frame = cv2.flip(frame, 1)

            # 计算视频高度和宽度
            frame_h, frame_w = frame.shape[:2]

            # 转为RGB
            img_cvt = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # 目标检测推理
            results = self.model(img_cvt)
            results_arr = results.pandas().xyxy[0].to_numpy()

            # 解析目标检测结果
            for item in results_arr:

                # 标签ID
                ret_label_id = item[-2]
                # 标签名称
                ret_label_text = item[-1]
                # 置信度
                ret_conf = item[-3]
                # 检测框坐标
                l,t,r,b = item[:4].astype('int')
                # 画矩形
                cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)
                # 绘制文字
                cv2.putText(frame,'{} {}%'.format(ret_label_text,round(ret_conf*100,2)),(l,t-10),cv2.FONT_ITALIC,1.5,(0,255,0),2)


            cv2.imshow('demo', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        cv2.destroyAllWindows()



if __name__ == '__main__':

    # 实例化
    test = yolo_test()
    test.demo()
    