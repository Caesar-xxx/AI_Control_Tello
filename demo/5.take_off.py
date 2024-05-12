"""
起飞Demo

"""

import ctypes
import cv2
import numpy as np
import time

# 多进程
from multiprocessing import Process, Value
# tello
from djitellopy import Tello

import torch


class Ai_tello:
    def __init__(self):
        # tello初始化
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()

        # ************************************ 绘制 相关 *********************************
        self.png_dict = {}
        # 获取
        self.getPngList()

        # ************************************ 结印动作相关变量*********************************

        # 加载 yolov5模型
        self.model = torch.hub.load(
            './yolov5', 'custom', path='./weights/pose.pt', source='local')
        # 置信度阈值
        self.model.conf = 0.5

        # 起飞指令激活时间
        self.take_off_time = None
        # 结印动作顺序
        self.yolo_action_seq = ['ani_1', 'ani_2','ani_3', 'ani_4', 'ani_5', 'ani_6']
        # 状态机，1表示当前动作已做完（击中）
        self.yolo_action_status = [0, 0, 0, 0, 0, 0]

        self.flight_control_command = Value(ctypes.c_int, -1)      # 飞行控制指令：-1 在地面

    def getPngList(self):
        '''
        读取PNG图片，追加进png_dict
        '''
        palm_action = {'ani_1': '巳', 'ani_2': '未', 'ani_3': '申',
            'ani_4': '亥', 'ani_5': '午', 'ani_6': '寅'}

        for name in palm_action.values():
            filename = './png_label/'+name+'.png'
            png_img = self.readPngFile(filename, 0.9)
            # {'寅':img numpy array,'hahha'}
            self.png_dict[name] = png_img

        print('PNG文字标签加载完毕')

    def readPngFile(self, fileName, scale=0.5):
        '''
        读取PNG图片
        '''
        # 解决中文路径问题
        png_img = cv2.imdecode(np.fromfile(fileName, dtype=np.uint8), -1)
        # 转为BGR，变成3通道
        png_img = cv2.cvtColor(png_img, cv2.COLOR_RGB2BGR)
        png_img = cv2.resize(png_img, (0, 0), fx=scale, fy=scale)
        return png_img  # img numpy array

    def addOverylay(self, frame, overlay, l, t):
        '''
        添加标签png覆盖
        '''
        # 解决l、t超界
        l = max(l, 0)
        t = max(t, 0)
        # 覆盖显示
        overlay_h, overlay_w = overlay.shape[:2]
        # 覆盖范围
        overlay_l, overlay_t = l, t
        overlay_r, overlay_b = (l + overlay_w), (overlay_t+overlay_h)
        # 遮罩
        overlay_copy = cv2.addWeighted(
            frame[overlay_t:overlay_b, overlay_l:overlay_r], 1, overlay, 20, 0)
            
        frame[overlay_t:overlay_b, overlay_l:overlay_r] = overlay_copy # numpy 的切片功能

    def flightControl(self):
        '''
        飞行控制进程调用函数
        '''

        # 启动另一个进程
        Process(target=ai_tello.cameraProcess).start()

        while True:

            if self.flight_control_command.value == 0:
                # 起飞指令
                print('take off')
                self.flight_control_command.value = 1
                # 起飞
                self.tello.takeoff()
                # 向上飞50CM
                self.tello.move_up(50)

            time.sleep(0.001)

    
    def cameraProcess(self):
        '''
        视频流处理：动作识别、绘制等
        '''
        
        # cap = cv2.VideoCapture(0)
        # 动作
        palm_action = {'ani_1':'巳','ani_2':'未','ani_3':'申','ani_4':'亥','ani_5':'午','ani_6':'寅'}
    

        while True:
            

            # 读取视频帧
            # ret,frame = cap.read()
            frame = self.tello.get_frame_read().frame

            frame = cv2.flip(frame, 1)

            # 转为RGB
            img_cvt = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # 已经起飞
            if self.take_off_time != None:
                print('已经起飞')
            else:
                # 未起飞
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
                
                    # 只绘制人
                    if ret_label_text =='person':
                        l,t,r,b = item[:4].astype('int')
                        cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)
                        cv2.putText(frame,'{}%'.format(round(ret_conf*100,2)),(l,t-10),cv2.FONT_ITALIC,1.5,(0,255,0),2)

                    # ani_1,ani_2....ani_6
                    # 结印动作，且置信度要求高一些
                    if 'ani_' in  ret_label_text and ret_conf >= 0.7:

                        l,t,r,b = item[:4].astype('int')
                        # 绘制
                        cv2.rectangle(frame,(l,t),(r,b),(0,255,20),2)
                        # 绘制动作中文png                        
                        label_zh = palm_action[ret_label_text]
                        # 拿到对应中文文字的数组图片
                        overlay = self.png_dict[label_zh]
                        # 覆盖绘制
                        self.addOverylay(frame,overlay,l,t-100)
                        cv2.putText(frame,'{}%'.format(round(ret_conf*100,2)),(l+80,t-20),cv2.FONT_ITALIC,1.5,(255,0,255),2)


                        # 状态机列表中第一个0的索引
                        first_0_index = next(i for i,x in enumerate(self.yolo_action_status) if x == 0 )
                        # 对应动作名 ['ani_1', 'ani_2','ani_3', 'ani_4', 'ani_5', 'ani_6']
                        check_action_name = self.yolo_action_seq[first_0_index]

                        # 动作匹配
                        if ret_label_text == check_action_name:
                            # 赋值1
                            self.yolo_action_status[first_0_index] = 1
                            # 检查是否完毕
                            if self.yolo_action_status == [1,1,1,1,1,1]:
                                # 发送起飞指令，设为此时
                                self.take_off_time = time.time()
                                print('动作全部匹配完毕，发送起飞指令')
                                # 允许起飞
                                self.flight_control_command.value = 0

                            else:
                                print('击中一个动作，当前列表为'+str(self.yolo_action_status))
                        else:
                            print('未击中动作，当前列表为'+str(self.yolo_action_status))
                
                    

            cv2.imshow('demo', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        cv2.destroyAllWindows()


if __name__ == '__main__':

    # 实例化
    ai_tello = Ai_tello()
    # 开启两个进程（防止飞机运动时，画面静止）
    ai_tello.flightControl()
    