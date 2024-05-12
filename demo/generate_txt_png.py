'''
生成字幕需要的中文PNG图片
'''

from PIL import Image,ImageDraw,ImageFont

def generate(name='你好',color_label='green'):

    filename = './png_label/'+name+'.png'
    # 如果不存在，先生成
    # 背景
    bg = Image.new("RGBA",(400,100),(0,0,0,0))
    # 添加文字
    d = ImageDraw.Draw(bg)
    font  = ImageFont.truetype('./fonts/MSYH.ttc',80,encoding="utf-8")

    if color_label == 'green':
        color = (0,255,0,255)
    else:
        color = (255,0,255,255)
        

    d.text((0,0),name,font=font,fill=color)
    # 保存
    bg.save(filename)

    print('ok: '+ name)


# palm_action = {'ani_1':'巳','ani_2':'未','ani_3':'申','ani_4':'亥','ani_5':'午','ani_6':'寅'}
# green_list = ['电量:','状态:','上下差:','偏航差:','前后差:']
# state_list ={-1:'准备中',0:'起飞',1:'高度偏航锁定',2:'向前',3:'向后',4:'向左',5:'向右',6:'追踪',7:'手掌降落',8:'立即降落'}


# for item in green_list:
#     generate(item,'green')        

# for item in state_list.values():
#     generate(item,'pink')       

# for item in palm_action.values():
#     generate(item,'pink')             

generate('计算机视觉','green')