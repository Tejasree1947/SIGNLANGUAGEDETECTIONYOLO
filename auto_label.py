import cv2
from cvzone.HandTrackingModule import HandDetector
from copy import deepcopy
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

train_label_path = "train_data/labels/train"
train_img_path = "train_data/images/train"
val_label_path = "train_data/labels/val"
val_img_path = "train_data/images/val"

label = 'H'
label_class = '16'

counter = 0

wait = 1
curr_time = time.time()
last_time = time.time()

n_cx = 0
n_cy = 0
n_w = 0
n_h = 0

while True:
    curr_time = time.time()
    if (curr_time - last_time < wait):
        continue
    last_time = curr_time
    success, plain_img = cap.read()
    # cv2.imwrite(f'{train_img_path}/{label}_{str(counter)}.jpg', plain_img)
    hands, img = detector.findHands(deepcopy(plain_img))
    # img = cv2.rectangle(img, (5, 5), (img.shape[1], img.shape[0]), (255, 0, 0), 1)
    if hands:
        if len(hands) == 1:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            cx, cy = hand['center']
            # print(img.shape)
            # print(x, y, w, h)
            n_w = w / img.shape[1]
            n_h = h / img.shape[0]
            n_cx = cx / img.shape[1]
            n_cy = cy / img.shape[0]
            # print(n_cx, n_cy, n_w, n_h)
            # add 40 to w and h
            n_w += 40 / img.shape[1]
            n_h += 40 / img.shape[0]

        else:
            hand1 = hands[0]
            hand2 = hands[1]
            if hand1['type'] == hand2['type']:
                continue
            x1, y1, w1, h1 = hand1['bbox']
            x2, y2, w2, h2 = hand2['bbox']
            x = 0
            y = 0
            w = 0
            h = 0
            if x1 < x2:
                x = x1 - 20
                w = w2 + (x2 - x1) + 20
            else:
                x = x2 - 20
                w = w1 + (x1 - x2) + 20
            if y1 < y2:
                y = y1 - 20
                h = h2 + (y2 - y1) + 20
            else:
                y = y2 - 20
                h = h1 + (y1 - y2) + 20
            cx = x + (w / 2)
            cy = y + (h / 2)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            n_w = w / img.shape[1]
            n_h = h / img.shape[0]
            n_cx = cx / img.shape[1]
            n_cy = cy / img.shape[0]
            # print(n_cx, n_cy, n_w, n_h)
            # add 20 more to w and h while writing to txt
            n_w += 20 / img.shape[1]
            n_h += 20 / img.shape[0]


    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        if n_w < 0:
            n_w = 0
        elif n_w > 1:
            n_w = 1
        if n_h < 0:
            n_h = 0
        elif n_h > 1:
            n_h = 1
        
        if counter < 81:
            f'{train_img_path}/{label}_{str(counter)}.jpg'
            cv2.imwrite(f'{train_img_path}/{label}_{str(counter)}.jpg', plain_img)
            with open(f'{train_label_path}/{label}_{str(counter)}.txt', 'w') as f_txt:
                f_txt.write(f'{label_class} {str(n_cx)} {str(n_cy)} {str(n_w)} {str(n_h)}')
        elif counter < 101:
            cv2.imwrite(f'{val_img_path}/{label}_{str(counter)}.jpg', plain_img)
            with open(f'{val_label_path}/{label}_{str(counter)}.txt', 'w') as f_txt:
                f_txt.write(f'{label_class} {str(n_cx)} {str(n_cy)} {str(n_w)} {str(n_h)}')

        print(counter)
