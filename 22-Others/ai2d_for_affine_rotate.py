"""
Script: ai2d_for_affine_rotate.py
脚本名称：ai2d_for_affine_rotate.py

Description:
    This script performs real-time affine rotation preprocessing on video frames 
    using the Ai2d module on an embedded system. A rotation matrix is applied 
    to each frame to achieve geometric transformation, and the result is displayed.

    The script utilizes the PipeLine module for frame acquisition and display 
    management. The Ai2d instance is configured with an affine matrix to apply 
    a rotation (e.g., 45 degrees). Output is converted to RGB888 for IDE preview.

脚本说明：
    本脚本在嵌入式系统上使用 Ai2d 模块对视频帧执行实时仿射旋转预处理。
    它为每帧图像应用旋转矩阵，实现几何变换，并将结果显示出来。

    脚本使用 PipeLine 模块进行图像采集与显示管理，配置 Ai2d 以执行指定角度
    （如 45 度）的旋转操作。输出图像被转换为 RGB888 格式用于 IDE 显示预览。

Author: Canaan Developer
作者：Canaan 开发者
"""



from libs.PipeLine import PipeLine
from libs.AI2D import Ai2d
from libs.Utils import *
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import gc,sys,os,time,math
import image

if __name__ == "__main__":
    # 添加显示模式，默认hdmi，可选hdmi/lcd/lt9611/st7701/hx8399,其中hdmi默认置为lt9611，分辨率1920*1080；lcd默认置为st7701，分辨率800*480
    display_mode="lcd"
    rgb888p_size=[512,512]
    pl=PipeLine(rgb888p_size=rgb888p_size,display_mode=display_mode)
    pl.create()
    display_size=pl.get_display_size()
    my_ai2d=Ai2d(debug_mode=0) #初始化Ai2d实例
    my_ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
    # 旋转角度
    angle = 45
    theta = math.radians(angle)  # 将角度转换为弧度
    cx,cy=100,100
    # 创建旋转矩阵
    affine_matrix = [
        math.cos(theta), -math.sin(theta), (1-math.cos(theta))*cx+math.sin(theta)*cy,
        math.sin(theta),  math.cos(theta), -math.sin(theta)*cx+(1-math.cos(theta))*cy
    ]

    # 设置仿射变换预处理
    my_ai2d.affine(nn.interp_method.cv2_bilinear,0, 0, 127, 1,affine_matrix)
    # 构建预处理过程
    my_ai2d.build([1,3,512,512],[1,3,512,512])
    while True:
        with ScopedTiming("total",1):
            img = pl.get_frame()            # 获取当前帧数据
            print(img.shape)                # 原图shape为[1,3,512,512]
            ai2d_output_tensor=my_ai2d.run(img) # 执行affine预处理，旋转
            ai2d_output_np=ai2d_output_tensor.to_numpy() # 类型转换
            print(ai2d_output_np.shape)        # 预处理后的shape为[1,3,512,512]
            # 使用transpose处理输出为HWC排布的np数据，然后在np数据上创建RGB888格式的Image实例用于在IDE显示效果
            shape=ai2d_output_np.shape
            ai2d_output_tmp = ai2d_output_np.reshape((shape[0] * shape[1], shape[2]*shape[3]))
            ai2d_output_tmp_trans = ai2d_output_tmp.transpose()
            ai2d_output_hwc=ai2d_output_tmp_trans.copy().reshape((shape[2],shape[3],shape[1]))
            out_img=image.Image(512, 512, image.RGB888,alloc=image.ALLOC_REF,data=ai2d_output_hwc)
            out_img.compress_for_ide()
            gc.collect()                    # 垃圾回收
    pl.destroy()                            # 销毁PipeLine实例
