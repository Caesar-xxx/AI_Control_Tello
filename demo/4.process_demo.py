'''
多进程示例Demo
'''

from multiprocessing import Process,Value
import time
import ctypes


class Demo:
    def __init__(self):
        # 初始化类属性
        self.share_value = 0
        # 初始化共享变量
        self.exchange_val = Value(ctypes.c_int, 0)


    def run1(self):
        # 启动子进程
        b = Process(target=self.run2) # 实例化process
        b.start()

        for i in range(3):
            time.sleep(1)
            print('from run 1, 变量分别是 {}, {} '.format(self.share_value,self.exchange_val.value))

    def run2(self):
        for i in range(3):
            print('from run 2')
            time.sleep(1)      
            # 累加  
            self.share_value +=2
            self.exchange_val.value +=2



if __name__ == '__main__': # 告诉系统这是主进程
    # 启动主进程
    demo = Demo()
    demo.run1() # 主进程设置为run1