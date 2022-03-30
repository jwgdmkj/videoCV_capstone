import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

filepath = 'data/test.mp4'
video= cv2.VideoCapture(filepath)

if not video.isOpened():
    print("Could not Open :", filepath)
    exit(0)

#불러온 비디오 파일의 정보 출력
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print("length :", length)
print("width :", width)
print("height :", height)
print("fps :", fps)

#프레임을 저장할 디렉토리를 생성
try:
    if not os.path.exists(filepath[:-4]):
        os.makedirs(filepath[:-4])
except OSError:
    print ('Error: Creating directory. ' +  filepath[:-4])

count = 0

curaverage = np.empty((height/16, width[0]/8))
print(curaverage.shape)
stride = [0, 0]
heightidx, widthidx = 0, 0

while(video.isOpened()):
    ret, image = video.read()

    # 앞서 불러온 fps 값을 사용하여 3초마다 추출
    if(int(video.get(3)) % fps == 0):
        while heightidx < height or widthidx < width:

            pass
        count += 1
    break
video.release()

'''
1. 이미지의 RGB 색상 분포도를 plot형식으로 만든다.
이 plot과, 다음 프레임의 plot을 분석해, 유의미한 차이가 있으면 하이라이트. 
이 threshold는 어떻게 정할 것이며, 무엇보다 두 개의 이미지 차이의 비율은 어떻게 구함?

2. pooling 방식으로, n*n 사이즈의 커널을 통과시켜, average pooling을 시전.
풀링시켜 생성한 하나의 이미지를 다음 프레임의 풀링시켜 획득한 이미지와 비교,
그 threshold가 일정 이상 차이가 있으면, 그것이 하이라이트
근데 이것 역시, threshold를 어떻게?
'''


'''
#cv2.imwrite(filepath[:-4] + "/frame%d.jpg" % count, image)
        #pil_image=Image.fromarray(image)
        planes = cv2.split(image)
        colors = ['b', 'g','r']

        fig, arr = plt.subplots(1, 2, figsize = (15,15))
        arr[0].imshow(image)
        for (plane, c) in zip(planes, colors):
            hist = cv2.calcHist([plane], [0], None, [256], [0, 256])
            plt.plot(hist, color=c)
        #plt.show()
        #print('Saved frame number :', str(int(video.get(1))))
'''
