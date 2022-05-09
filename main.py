import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

filepath = 'C:/Users/is7se/Desktop/코드용/videoCV/data_2/test.mp4'
filepath2 = './'
video= cv2.VideoCapture(filepath)

if not video.isOpened():
    print("Could not Open :", filepath)
    exit(0)

#불러온 비디오 파일의 정보 출력
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
alllength = length//fps # 총 길이

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
'''
while(video.isOpened()):
    ret, image = video.read()
    if image is None:
        break
    #fig, arr = plt.subplots(1, 2, figsize = (15,15))
    #arr[0].imshow(image)
    # 앞서 불러온 fps 값을 사용하여 1초마다 추출
    if(int(video.get(1)) % 60 == 0):
        # ----- CNN마냥 필터를 돌아다니며 추출 -----
        while heightidx + height//10 < (2 * height) // 3:
            lstfornowscreen = []
            while widthidx + width//16 < (width * 2) // 3:
                # 일정 부분을 잘라냄.
                curaverage = image[heightidx : heightidx+(height//10),\
                         widthidx : widthidx+ (width//16),:].copy()
                sumnp = np.zeros((1, 3))
                for i in range(len(curaverage)):
                    for j in range(len(curaverage[0])):
                        sumnp += curaverage[i][j]
                sumnp /= len(curaverage) * len(curaverage[0]) # 특정 커널의 평균
                lstfornowscreen.append(sumnp)
                widthidx += (int)(width//16) # widthidx 옮기기

            nowscreen.append(lstfornowscreen)
            heightidx += (int)(height//10) #heightidx 옮기기
            widthidx = 0
        # ----- 필터 돌아다니며 추출하기 끝 -----

        # ----- 이전 프레임(prevscreen)과 비교하면서, 차이를 비교 -----
        if len(prevscreen) > 0:
            pass
        #국소적 부위에 눈에 띌만한 차이가 있고, 그것이 여러 곳에서 관측된다면 하이라이트?
        #단, 모든 부위에 변화가 관찰된다면 이는 배경이 바뀐 것일 가능성이 큼.
        #따라서 몇 개의 범위가 바껴야 하이라이트로 간주할지, 얼마나 바껴야 할지 임계점은?

        #print(second)
        #print(nowscreen)
        #print('-------')
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

print(widthidx, width//12, width*2//3)
print(heightidx, height//10, 2*height//3)
heightcnt = (int)(((2*height//3) - heightidx) / (height//10))
widthcnt = (int)(((2*width//3) - widthidx) / (width//12))

# (640, 800, 960, 1120) - width / (180, 288, 396, 504, 612) - height
lstH = []

while(video.isOpened()):
    ret, image = video.read()
    if image is None:
        break

    # 앞서 불러온 fps 값을 사용하여 0.5초마다 추출
    if(int(video.get(1)) % 30 == 0):
        print(second, end = ' ')
        averH = 0 # 합산시킬 H

        lstfornowscreen = [] #같은 프레임 내의 이미지
        while heightidx + height//10 <= (2 * height) // 3:
            while widthidx + width//12 <= (width * 2) // 3:
                # 일정 부분을 잘라냄.
                curaverage = image[heightidx : heightidx+(height//10),\
                         widthidx : widthidx+ (width//12),:].copy()
                averH = 0
                
                for i in range(len(curaverage)):
                    for j in range(len(curaverage[0])):
                        newR, newG, newB = (curaverage[i][j][0]/255),\
                                           (curaverage[i][j][1]/255),\
                                           (curaverage[i][j][2]/255)
                        Min = min(newR, newG, newB)
                        Max = max(newR, newG, newB)
                        del_max = Max - Min
                        H = 0

                        #H값 추출
                        if(del_max == 0):
                            H = 0
                        else:
                            if newR == Max:
                                H = ((((Max - newB)/6) + del_max/2)) / \
                                    ((((Max - newG)/6)+(del_max/2))/del_max)
                            elif newG == Max:
                                H = (1/3)+((((Max-newR)/6)+(del_max/2)) / del_max) \
                                    - ((((Max - newB)/6)+(del_max/2))/del_max)
                            elif newB == Max:
                                H = (2/3)+((((Max-newG)/6)+(del_max/2)) / del_max) - \
                                    ((((Max - newR)/6)+(del_max/2))/del_max)
                            if H<0:
                                H += 1
                            if H>1:
                                H -= 1

                        averH += H
                #print('widthidx', end = ' ')
                #print(widthidx)
                averH /= len(curaverage) * len(curaverage[0]) # 특정 커널의 평균
                lstfornowscreen.append(averH)
                widthidx += (int)(width//12) # widthidx 옮기기
                # --------- 가로 한 줄 끝 --------------#
            # ---------세로로 하나 내리기 ------------#
            #print('height', end= ' ')
            #print(heightidx)
            heightidx += (int)(height//10) #heightidx 옮기기
            widthidx = width//3
            # ------------한 화면 끝 --------------#
        lstH.append(lstfornowscreen)
        heightidx = height//6 # 기준이 되는 지점
        second += 1
        # ---------- 다음 초로 넘어가기 ----------#
        #if second == 240:
        #    break

video.release()
print(lstH)
stdList = (np.std(lstH, axis = 0))
print(stdList)

# 표준편차를 기준으로 탐색할 범위 지정
# 평균값 +- 표준편차
avgCntStart, avgCntEnd = [], []
for i in range(len(stdList)): # 총 격자 개수
    tmp = 0
    for j in range(len(lstH)): # 총 날짜 개수
        tmp += lstH[j][i]
    avgCntStart.append((tmp / second) - stdList[i])
    avgCntEnd.append((tmp / second) + stdList[i])

print(avgCntStart)
print(avgCntEnd)
'''
탐색범위를 돌며, 영역 중 떨어지는 애를 구한다. 한 프레임 내에 10~50% 정도가 떨어지는 애들이면, 개 선택
'''

acceptFrameList = []
# 탐색범위 내의 프레임만 필터링

for i in range(len(lstH)): # 총 날짜 개수
    tmparr=[]
    for j in range(len(lstH[i])): # 총 격자 개수
        if (lstH[i][j] < avgCntStart[j] and lstH[i][j] > avgCntStart[j] - stdList[j]) \
        or (lstH[i][j] > avgCntEnd[j] and lstH[i][j] < avgCntEnd[j] + stdList[j]) :
            tmparr.append(j)

    # 편차가 있는 곳이 1 이상 1/2 미만인 경우
    if len(tmparr) > (heightcnt * widthcnt) // 3 \
            and len(tmparr) < (heightcnt * widthcnt) // 2:
        acceptFrameList.append(i/2)

print(acceptFrameList)

'''
highlightArr = []
startidx, endidx = 0, 0
# 보정 과정, 2.5초 지속되는 경우 하이라이트로 지정
frameidx = 0
consec = 0
for i in range(len(acceptFrameList)-1):
   if acceptFrameList[i+1] == acceptFrameList[i]+1:
       # 만약 연속이 시작될 경우(그전까지 0이었다면)
       if consec == 0:
           startidx = acceptFrameList[i] # 시작 idx 설정
       consec += 1 # 연속 = 1 증가

       if i == len(acceptFrameList)-2 and consec >= 3 and acceptFrameList[i]+1 == acceptFrameList[i+1]:
            endidx = acceptFrameList[i]+1
            highlightArr.append((startidx, endidx))
            consec, startidx, endidx = 0, 0, 0
   else :
       if consec >= 3: # 4번 이상 충분히 유지되었다면, endidx를 설정
           endidx = acceptFrameList[i]
           highlightArr.append((startidx, endidx))
       consec, startidx, endidx = 0, 0, 0

print(highlightArr)
# 각 구간 별 차이가 5초 이하라면, 연결
for i in range(len(highlightArr)-1):
    if highlightArr[i+1][0] - highlightArr[i][1] < 4:
        newhighlight = (highlightArr[i][0], highlightArr[i+1][1])
        del highlightArr[i+1], highlightArr[i]
        highlightArr.append(newhighlight)
print(highlightArr)
'''
# # 수집한 하이라이트 구간을 원본영상에서 자름
# video.set(cv2.CV_CAP_POS_FRAMES, acceptFrameList[0])
#
# parts = [(15, 30), (50, 79)]
# cap = cv2.VideoCapture(filepath)
# ret, frame = cap.read()
# h, w, _ = frame.shape
#
# # Define a fourcc (four-character code), and define a list of video writers;
# # 3rd 파라미터: fps, 4th 파라미터: frame Size
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# writers = [cv2.VideoWriter(f"part{start}-{end}.avi", fourcc, 20.0, (w, h)) \
#            for start, end in parts]
#
# # Define a while loop, but before that, define a variable to
# # keep track of which frame the while loop is at in the capture device:
# f = 0
# while ret: # ret == False시, 중지
#     f += 1 # f == frame
#
#     # Using a for loop inside the while loop,
#     # loop through the start and end frames of the parts,
#     # using the enumerate method to allow the program
#     # to access the index of the parts each iteration is at when needed.
#     # If the variable defined before the while loop is between (inclusive)
#     # the start and end of the part, write that frame to the
#     # corresponding video writer (using the index provided
#     # by the enumerate method):
#     for i, part in enumerate(parts):
#         start, end = part
#         if start <= f <= end:
#             writers[i].write(frame)
#     ret, frame = cap.read()
#
# for writer in writers:
#     writer.release()
#
# cap.release()
#
# # 자른 영상을 영상리스트(txt)를 기준으로 자르기


'''
        for i in range(height//6, (height*4)//6):
            tmparr = []
            for j in range(width//3, (width*2)//3):
                newR, newG, newB = (image[i][j][0]/255),(image[i][j][1]/255),(image[i][j][2]/255)
                Min = min(newR, newG, newB)
                Max = max(newR, newG, newB)
                del_max = Max - Min
                H = 0

                #H값 추출
                if(del_max == 0):
                    H = 0
                else:
                    if newR == Max:
                        H = ((((Max - newB)/6) + del_max/2)) / ((((Max - newG)/6)+(del_max/2))/del_max)
                    elif newG == Max:
                        H = (1/3)+((((Max-newR)/6)+(del_max/2)) / del_max) - ((((Max - newB)/6)+(del_max/2))/del_max)
                    elif newB == Max:
                        H = (2/3)+((((Max-newG)/6)+(del_max/2)) / del_max) - ((((Max - newR)/6)+(del_max/2))/del_max)
                    if H<0:
                        H += 1
                    if H>1:
                        H -= 1
                tmparr.append((H, 0, 1))
                averH += H
            prevscreen.append(tmparr)
        averH /= (width * height) # 한 프레임의 H값 평균
        lstH.append(averH) # 이걸 리스트에 넣기

        second += 1
    #if second == 80:
    #    break
'''
