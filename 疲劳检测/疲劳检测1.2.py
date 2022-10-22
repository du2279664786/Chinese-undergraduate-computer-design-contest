import tkinter
import matplotlib

# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import numpy as np

import cv2
import matplotlib.pyplot as plt
import paddlehub as hub

def fatigue_testing(img):
    face_landmark = hub.Module(name="face_landmark_localization")
    # result = face_landmark.keypoint_detection(images=[cv2.imread('work/1.png')], use_gpu=False)
    result = face_landmark.keypoint_detection(images=[img[1]], use_gpu=False)
    face_landmark = result[0]['data'][0]

    # 左眼纵线坐标
    l1 = (np.array(face_landmark[39]) + np.array(face_landmark[38])) / 2
    l2 = (np.array(face_landmark[41]) + np.array(face_landmark[42])) / 2
    # 左眼纵线坐标欧氏距离
    L1 = ((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2) ** 0.5
    # 左眼横线坐标欧氏距离
    c1 = face_landmark[37]
    c2 = face_landmark[40]
    L2 = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    # 右眼纵线坐标
    r1 = (np.array(face_landmark[44]) + np.array(face_landmark[45])) / 2
    r2 = (np.array(face_landmark[48]) + np.array(face_landmark[47])) / 2
    # 右眼纵线坐标欧氏距离
    R1 = ((r1[0] - r2[0]) ** 2 + (r1[1] - r2[1]) ** 2) ** 0.5
    # 右眼横线坐标欧氏距离
    d1 = face_landmark[43]
    d2 = face_landmark[46]
    R2 = ((d1[0] - d2[0]) ** 2 + (d1[1] - d2[1]) ** 2) ** 0.5

    # 计算域值
    LEAR = L1 / (L2 * 2)
    REAR = R1 / (R2 * 2)
    ear = (LEAR + REAR) / 2

    # 在图片上标注
    # 左眼坐标
    LEFTEyeArea = np.array([
        face_landmark[38], face_landmark[39],
        face_landmark[42], face_landmark[41],
        face_landmark[37], face_landmark[40],
    ], dtype='float')
    # 右眼坐标
    RIGHTEyeArea = np.array([
        face_landmark[44], face_landmark[45],
        face_landmark[48], face_landmark[47],
        face_landmark[43], face_landmark[46],
    ], dtype='float')
    # 判断是否疲劳
    img1 = img[1]
    if ear > 1.75:
        cv2.putText(img1, "WARNING!!!YOU ARE TIRED!", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(img1, "everything is ok~", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # 转换一下颜色通道
    b, g, r = cv2.split(img1)
    img_rgb = cv2.merge([r, g, b])
    return img_rgb


cap = cv2.VideoCapture(0)
if (cap.isOpened()):  # 视频打开成功
    while (True):
        # ret, frame = cap.read()  # 读取一帧
        # result = mask_detecion(frame)

        img = cap.read()
        img_rgb = fatigue_testing(img)
        cv2.imshow('testing', img_rgb)
        if cv2.waitKey(1) & 0xFF == 27:  # 按下Esc键退出
            break
else:
    print('open video/camera failed!')
cap.release()
cv2.destroyAllWindows()

# plt.figure(figsize=(20, 10))
# plt.imshow(img_rgb)
# plt.axis('off')
# plt.show()