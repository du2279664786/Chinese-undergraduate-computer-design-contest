# encoding=utf-8
import winsound
import cv2
import numpy as np
import os
import time
import datetime

a = datetime.datetime.now()


def yolo_detect(pathIn='',
                pathOut=None,
                label_path='E:/cfg/coco.names',
                config_path='E:/cfg/yolov3.cfg',
                weights_path='E:/cfg/yolov3.weights',
                confidence_thre=0.5,
                nms_thre=0.3,
                jpg_quality=80):
    '''
    pathIn：原始图片的路径
    pathOut：结果图片的路径
    label_path：类别标签文件的路径
    config_path：模型配置文件的路径
    weights_path：模型权重文件的路径
    confidence_thre：0-1，置信度（概率/打分）阈值，即保留概率大于这个值的边界框，默认为0.5
    nms_thre：非极大值抑制的阈值，默认为0.3
    jpg_quality：设定输出图片的质量，范围为0到100，默认为80，越大质量越好
    '''

    # 加载类别标签文件
    LABELS = open(label_path).read().strip().split("\n")
    nclass = len(LABELS)

    # 为每个类别的边界框随机匹配相应颜色
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(nclass, 3), dtype='uint8')

    # 载入图片并获取其维度
    base_path = os.path.basename(pathIn)
    img = cv2.imread(pathIn)
    (H, W) = img.shape[:2]

    # 加载模型配置和权重文件
    print('从硬盘加载YOLO......')
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # 获取YOLO输出层的名字
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # 将图片构建成一个blob，设置图片尺寸，然后执行一次
    # YOLO前馈网络计算，最终获取边界框和相应概率
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # 显示预测所花费时间
    print('YOLO模型花费 {:.2f} 秒来预测一张图片'.format(end - start))

    # 初始化边界框，置信度（概率）以及类别
    boxes = []
    confidences = []
    classIDs = []

    # 迭代每个输出层，总共三个
    for output in layerOutputs:
        # 迭代每个检测
        for detection in output:
            # 提取类别ID和置信度
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # 只保留置信度大于某值的边界框
            if confidence > confidence_thre:
                # 将边界框的坐标还原至与原图片相匹配，记住YOLO返回的是
                # 边界框的中心坐标以及边界框的宽度和高度
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # 计算边界框的左上角位置
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # 更新边界框，置信度（概率）以及类别
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # 使用非极大值抑制方法抑制弱、重叠边界框
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thre, nms_thre)

    # 确保至少一个边界框
    if len(idxs) > 0:
        # 迭代每个边界框
        for i in idxs.flatten():
            # 提取边界框的坐标
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # 绘制边界框以及在左上角添加类别标签和置信度
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.3f}'.format(LABELS[classIDs[i]], confidences[i])
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x, y - text_h - baseline), (x + text_w, y), color, -1)
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # return img
    # 输出结果图片
    if pathOut is None:
        cv2.imwrite('with_box_' + base_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    else:
        cv2.imwrite(pathOut, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])


camera = cv2.VideoCapture(0)  # 定义摄像头对象，其参数0表示第一个摄像头（自带摄像头），还可以是视频的路径
if camera is None:
    # 如果摄像头打开失败，则输出提示信息
    print('please connect the camera')
    exit()

fps = 80  # 帧率
pre_frame = None  # 总是取前一帧做为背景（不用考虑环境影响）


def rec_time():
    year = str(datetime.datetime.now().year)
    month = str(datetime.datetime.now().month)
    day = str(datetime.datetime.now().day)
    hour = str(datetime.datetime.now().hour)
    minute = str(datetime.datetime.now().minute)
    second = str(datetime.datetime.now().second)
    i = year + '年' + month + '月' + day + '日' + hour + "点" + minute + '分' + second + '秒'
    return i


i = 1

while True:
    start = time.time()
    # 读取视频流
    res, cur_frame = camera.read()
    # ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False
    # cur_frame是每一帧的图像，是个三维矩阵
    if res != True:
        break
    end = time.time()

    seconds = end - start
    if seconds < 1.0 / fps:
        time.sleep(1.0 / fps - seconds)

    cv2.namedWindow('img', 0);
    # cv2.imshow('img', cur_frame)

    # 检测如何按下Q键，则退出程序
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break
    # 转灰度图
    gray_img = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    # 将图片缩放
    gray_img = cv2.resize(gray_img, (500, 500))
    # 用高斯滤波进行模糊处理
    gray_img = cv2.GaussianBlur(gray_img, (21, 21), 0)

    # 如果没有背景图像就将当前帧当作背景图片
    if pre_frame is None:
        pre_frame = gray_img
    else:
        # absdiff把两幅图的差的绝对值输出到另一幅图上面来
        img_delta = cv2.absdiff(pre_frame, gray_img)

        # threshold阈值函数(原图像应该是灰度图,对像素值进行分类的阈值,当像素值高于（有时是小于）
        # 阈值时应该被赋予的新的像素值,阈值方法)
        thresh = cv2.threshold(img_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # 膨胀图像
        thresh = cv2.dilate(thresh, None, iterations=2)

        # findContours检测物体轮廓(寻找轮廓的图像,轮廓的检索模式,轮廓的近似办法)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            # 灵敏度
            if cv2.contourArea(c) < 1000:  # 1000为阈值
                continue
            else:
                # 框选移动部分
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(cur_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                print('--------------------------------------------------------')
                print('时间：' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '有物体移动')
                pa = rec_time()
                pathin = 'E:/yoloSrc/messigray' + str(i) + '出现物体.png'

                cv2.imwrite(pathin, cur_frame)
                # print(1)
                pathout = 'E:/yoloSrc/messigray' + str(i) + '出现物体.png'

                yolo_detect(pathin, pathout)
                print(1)
                # cv2.resizeWindow("resized", 640, 480)
                img = cv2.imread(pathout, flags=1)
                cv2.imshow('img', img)
                # pre_frame = pathout
                i += 1
                flag = True
                if flag == True:
                    winsound.Beep(600, 1200)
                break
        # 显示
        # img = cv2.imread(pathout,flags=1)
        # cv2.imshow('img', img)
        # print(cur_frame)
        pre_frame = gray_img

# release()释放摄像头
camera.release()
# destroyAllWindows()关闭所有图像窗口
cv2.destroyAllWindows()
