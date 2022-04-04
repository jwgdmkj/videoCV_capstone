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

#curaverage = np.empty(((int)(height/8), (int)(width/16), 3))
stride = [0, 0]
heightidx, widthidx = height//6, width//3 # 기준이 되는 지점
#print(4*heightidx, 2*widthidx)
second = 0

prevscreen, nowscreen = [], []

while(video.isOpened()):
    ret, image = video.read()
    #fig, arr = plt.subplots(1, 2, figsize = (15,15))
    #arr[0].imshow(image)
    # 앞서 불러온 fps 값을 사용하여 1초마다 추출
    if(int(video.get(1)) % 60 == 0):
        # ----- CNN마냥 필터를 돌아다니며 추출 -----
        while heightidx + height//20 < (2 * height) // 3:
            lstfornowscreen = []
            while widthidx + width//32 < (width * 2) // 3:
                # 일정 부분을 잘라냄.
                curaverage = image[heightidx : heightidx+(height//10),\
                         widthidx : widthidx+ (width//16),:].copy()
                sumnp = np.zeros((1, 3))
                for i in range(len(curaverage)):
                    for j in range(len(curaverage[0])):
                        sumnp += curaverage[i][j]
                sumnp /= len(curaverage) * len(curaverage[0]) # 특정 커널의 평균
                lstfornowscreen.append(sumnp)
                widthidx += (int)(width//32) # widthidx 옮기기

            nowscreen.append(lstfornowscreen)
            heightidx += (int)(height//20) #heightidx 옮기기
            widthidx = 0
        # ----- 필터 돌아다니며 추출하기 끝 -----

        # ----- 이전 프레임(prevscreen)과 비교하면서, 차이를 비교 -----
        '''
        국소적 부위에 눈에 띌만한 차이가 있고, 그것이 여러 곳에서 관측된다면 하이라이트?
        단, 모든 부위에 변화가 관찰된다면 이는 배경이 바뀐 것일 가능성이 큼.
        따라서 몇 개의 범위가 바껴야 하이라이트로 간주할지, 얼마나 바껴야 할지 임계점은?
        '''
        print(second)
        print(nowscreen)
        print('-------')
        # ----- 차이 비교 끝, now와 prev 갱신 필요 -----
        prevscreen = nowscreen.copy()
        nowscreen = []
        second += 1

        heightidx, widthidx = height//6, width//3 # 기준이 되는 지점

        #arr[1].imshow(curaverage)
    #break
    #plt.show()
video.release()

'''
1. 이미지의 RGB 색상 분포도를 plot형식으로 만든다.
이 plot과, 다음 프레임의 plot을 분석해, 유의미한 차이가 있으면 하이라이트. 
이 threshold는 어떻게 정할 것이며, 무엇보다 두 개의 이미지 차이의 비율은 어떻게 구함?

2. pooling 방식으로, n*n 사이즈의 커널을 통과시켜, average pooling을 시전.
풀링시켜 생성한 하나의 이미지를 다음 프레임의 풀링시켜 획득한 이미지와 비교,
그 threshold가 일정 이상 차이가 있으면, 그것이 하이라이트
근데 이것 역시, threshold를 어떻게?

3. 캐릭터와 일정 범위를 추출
색깔 변화가 일정 비율을 넘는다 --> 배경이 바뀌는 것으로 간주
일정비율 내일 경우 --> grid 간의 색갈 찰르 비교, 같다면 배경바뀌는 도중임.
화면길이 가로 20cm기준 캐릭터는 9.5~10.5 // 세로는 12cm기준 4.5~6.5
또한, 프레임 간 간격은 1초로도 충분? 하이라이트 특징 상 앞뒤는 빡빡하게 안잡아도 됨
그럼 비교할 분할영역은 세로 2~8?(0~12중) 가로 6,7,8~13,14 즈음?
그럼 1/6(0.17)부터 2/3(0.67)까지, 1/3(0.34)부터 13/20(0.68)까지인건데

일정 범위 내의 픽셀이 계속 유지되면, 이것이 평상시라 가정. 이는 갱신될 수 있다.
이후에 변화가 생길 때, 하이라이트든 뭐든 발생한 것이라 가정.
다시 이 평상시로 픽셀이 돌아오는 경우가 하이라이트가 제거된 것.

while(video.isOpened()):
    ret, image = video.read()
    #print(image[500][900])
    #fig, arr = plt.subplots(1, 2, figsize = (15,15))
    #arr[0].imshow(image)
    # 앞서 불러온 fps 값을 사용하여 3초마다 추출
    if(int(video.get(3)) % fps == 0):
        while heightidx + height//20 < height:
            while widthidx + width//32 < width:
                # curaverae 사이즈 : (135,120)
                curaverage = image[heightidx:heightidx+(int)(height//10),\
                         widthidx:widthidx+(int)(width//16),:].copy()
                #curaverage = image[750:755, 700:704, :].copy()
                #print(len(curaverage), len(curaverage[0]), end='\t')
                sunmp = np.zeros((1, 3))
                for i in range(len(curaverage)):
                    for j in range(len(curaverage[0])):
                        #print(len(curaverage), len(curaverage[0]), end='\t')
                        sunmp += curaverage[i][j]

                #print(curaverage)
                #print('------------------')
                sumnp = sunmp/((int)(width/16) * (int)(height/10))
                #print(sumnp, end='\t')
                #print(heightidx, widthidx)
                widthidx += (int)(width//32)
                #arr[1].imshow(curaverage)
                count += 1
                #break
            heightidx += (int)(height//20)
            widthidx = 0
            #print('new line')
            #break
    break


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
