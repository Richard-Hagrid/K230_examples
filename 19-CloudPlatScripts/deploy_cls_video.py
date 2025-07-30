# -*- coding: utf-8 -*-
'''
Script: deploy_cls_video.py
脚本名称：deploy_cls_video.py

Description:
    This script runs a real-time image classification application on an embedded device.
    It uses a pipeline to capture video frames, performs inference using a pre-trained Kmodel,
    and displays the classification result on screen.

    The model configuration is loaded from a JSON config file.

脚本说明：
    本脚本在嵌入式设备上运行实时图像分类应用。它通过捕获视频帧，使用预训练的 Kmodel 进行推理，并在屏幕上显示分类结果。

    模型配置文件通过 JSON 文件从 Canaan 在线训练平台加载。

Author: Canaan Developer
作者：Canaan 开发者
'''


import os, gc
from libs.PlatTasks import ClassificationApp
from libs.PipeLine import PipeLine
from libs.Utils import *

# Set display mode: options are 'hdmi', 'lcd', 'lt9611', 'st7701', 'hx8399'
# 'hdmi' defaults to 'lt9611' (1920x1080); 'lcd' defaults to 'st7701' (800x480)
display_mode = "lt9611"

# Define the input size for the RGB888P video frames
rgb888p_size = [1280, 720]

# Set root directory path for model and config
root_path = "/sdcard/mp_deployment_source/"

# Load deployment configuration
deploy_conf = read_json(root_path + "/deploy_config.json")
kmodel_path = root_path + deploy_conf["kmodel_path"]              # KModel path
labels = deploy_conf["categories"]                                # Label list
confidence_threshold = deploy_conf["confidence_threshold"]        # Confidence threshold
model_input_size = deploy_conf["img_size"]                        # Model input size
num_classes = deploy_conf["num_classes"]                          # Number of classes

# Inference configuration
inference_mode = "video"                                          # Inference mode: 'video'
debug_mode = 0                                                    # Debug mode flag

# Create and initialize the video/display pipeline
pl = PipeLine(rgb888p_size=rgb888p_size, display_mode=display_mode)
pl.create()
display_size = pl.get_display_size()

# Initialize classification application
cls_app = ClassificationApp(inference_mode,kmodel_path,labels,model_input_size,confidence_threshold,rgb888p_size,display_size,debug_mode=debug_mode)

# Configure preprocessing for the model
cls_app.config_preprocess()

# Main loop: capture, run inference, display results
while True:
    with ScopedTiming("total", 1):
        img = pl.get_frame()                          # Capture current frame
        res = cls_app.run(img)                        # Run inference
        cls_app.draw_result(pl.osd_img, res)          # Draw classification result
        pl.show_image()                               # Show result on display
        gc.collect()                                  # Run garbage collection to free memory

# Cleanup: These lines will only run if the loop is interrupted (e.g., by an IDE break or external interruption)
cls_app.deinit()                                      # De-initialize classification app
pl.destroy()                                          # Destroy pipeline instance



