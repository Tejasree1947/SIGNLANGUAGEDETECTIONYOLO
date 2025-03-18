import cv2
import glob
import imgaug.augmenters as iaa
from cvzone.HandTrackingModule import HandDetector
from copy import deepcopy


detector = HandDetector(maxHands=2)
print('detector init')

images = []
prefix = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H",
          "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
image_path = glob.glob("train_data/images/train/*.jpg")
for img_path in image_path:
    img = cv2.imread(img_path)
    images.append(img)
print('image path init')

augmentation = iaa.Sequential([
    # iaa.Rotate((-15, 15)),
    # iaa.Multiply((0.7, 1.5)),
    iaa.GaussianBlur(sigma=(0, 0.5))
])

augmented_images = augmentation(images=images)
print('augement init')
print(len(augmented_images))

n_cx = 0
n_cy = 0
n_w = 0
n_h = 0


pref_count = 0
for pref in prefix:
    count = 1
    detected = False
    for i in range(80):
        hands, img2 = detector.findHands(deepcopy(augmented_images[80 * pref_count + i]))
    # img = cv2.rectangle(img, (5, 5), (img.shape[1], img.shape[0]), (255, 0, 0), 1)
        if hands:
            detected = True
            if len(hands) == 1:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                cx, cy = hand['center']
                # print(img.shape)
                # print(x, y, w, h)
                n_w = w / img2.shape[1]
                n_h = h / img2.shape[0]
                n_cx = cx / img2.shape[1]
                n_cy = cy / img2.shape[0]
                # print(n_cx, n_cy, n_w, n_h)
                # add 40 to w and h
                n_w += 40 / img2.shape[1]
                n_h += 40 / img2.shape[0]
                

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
                img2 = cv2.rectangle(
                    img2, (x, y), (x + w, y + h), (255, 0, 0), 1)
                n_w = w / img2.shape[1]
                n_h = h / img2.shape[0]
                n_cx = cx / img2.shape[1]
                n_cy = cy / img2.shape[0]
                # print(n_cx, n_cy, n_w, n_h)
                # add 20 more to w and h while writing to txt
                n_w += 20 / img2.shape[1]
                n_h += 20 / img2.shape[0]
        else:
            print('skipped', pref_count, i)

        if count <= 80:
            if detected:
                filename = 'save2/aug3{}_{}.jpg'.format(pref, count)
                cv2.imwrite(filename, augmented_images[80 * pref_count + i])
                with open(f'save3/aug3{pref}_{count}.txt', 'w') as f_txt:
                    f_txt.write(
                        f'{pref} {str(n_cx)} {str(n_cy)} {str(n_w)} {str(n_h)}')
                count += 1
                detected = False
                if count/40 == 2:
                    print("Done")
                    print(80 * pref_count + i)
        else:
            break
    pref_count += 1
