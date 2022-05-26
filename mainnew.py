import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

filepath = 'C:/Users/is7se/Desktop/코드용/videoCV/data_2/Lol_2.mp4'
filepath2 = './'
video= cv2.VideoCapture(filepath)

if not video.isOpened():
    print("Could not Open :", filepath)
    exit(0)

import time
starttime = time.time()

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

# 기준이 되는 지점 : 가로 1280 기준 224 : 양옆 1/5만큼 비우기. 24/40를 4등분, 총너비 대비 (실수형)
# 위는 720 기준 7/40(126px) 세로는 13/40(234px) 비우기. 20/40을 4등분, 총 높이 대비 1/8
# (256 448 640 832 1024) (126 216 306 396 486)
#heightidx, widthidx = height//6, width//3 # 기준이 되는 지점
heightidx, widthidx = (7*height)//40, width//5 # 기준점
heightmax, widthmax = (27*height)//40, (width*4)//5 # 탐색범위 끝
heightsplitter, widthsplitter = 4, 4 # 분할 개수.

# 0.5초 별 체크할 이미지의 사이즈
heightsz, widthsz = height//2, (width*3)//5

# 각 격자의 너비/높이별 개수
heightcnt = (int)(((27*height//40) - heightidx) / (height//heightsplitter))
widthcnt = (int)(((5*width//6) - widthidx) / (width//widthsplitter))

print(widthidx, widthmax, heightsz, heightsz//heightsplitter)
print(heightidx, heightmax, widthsz, widthsz//widthsplitter)
second = 0

#heightcnt = (int)(((2*height//3) - heightidx) / (height//10)) 
#widthcnt = (int)(((2*width//3) - widthidx) / (width//12)) 
# (640, 800, 960, 1120) - width / (180, 288, 396, 504, 612) - height
# 640 ~ 1280 & 180 ~ 720
# 160 = width(1920)/12, 108 = height(1080)/10

prevscreen, nowscreen = [], []

#heightsplitter, widthsplitter = 10, 12
lstH = []

while(video.isOpened()):
    ret, image = video.read()
    if image is None:
        break

    # 30 : 앞서 불러온 fps 값을 사용하여 0.5초마다 추출
    if(int(video.get(1)) % 30 == 0):
        print(second, end = ' ')
        averH = 0 # 합산시킬 H
        #cv2.imshow('now' + str(second), image)

        lstfornowscreen = [] #같은 프레임 내의 이미지
        # (256 448 640 832 1024) (126 216 306 396 486)
        #print(heightidx, heightsz, heightsplitter, heightsz//heightsplitter)
        #print(widthidx, widthsz, widthsplitter, widthsz//widthsplitter)
        while heightidx + heightsz//heightsplitter <= heightmax : #(2 * height) // 3:
            while widthidx + widthsz//widthsplitter <= widthmax : #(width * 2) // 3:
                # 일정 부분을 잘라냄.
                curaverage = image[heightidx : heightidx+(heightsz//heightsplitter),\
                         widthidx : widthidx+ (widthsz//widthsplitter),:].copy()
                averH = 0
                #cv2.imshow(str(second) + ' ' + str(heightidx) + ' ' + str(widthidx), curaverage)

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
                widthidx += (int)(widthsz//widthsplitter) # widthidx 옮기기
                # --------- 가로 한 줄 끝 --------------#
            # ---------세로로 하나 내리기 ------------#
            #print('height', end= ' ')
            #print(heightidx)
            heightidx += (int)(heightsz//heightsplitter) #heightidx 옮기기
            widthidx = width//5 # 기준 되는 지점
            # ------------한 화면 끝 --------------#
        #print(lstfornowscreen)
        lstH.append(lstfornowscreen)
        heightidx = (7*height)//40 # 기준이 되는 지점
        second += 1
        #cv2.waitKey(0)
        # ---------- 다음 초로 넘어가기 ----------#
        if second == 600:
            break

video.release()
#print('list of H is ', end='\t')
#print(lstH)
stdList = (np.std(lstH, axis = 0))  # 격자 별 분산 리스트
#print(stdList)

# -------------표준편차를 기준으로 탐색할 범위 지정----------------
# 평균값 +- 표준편차
avgCntStart, avgCntEnd = [], []
for i in range(len(stdList)): # 총 격자 개수
    tmp = 0
    for j in range(len(lstH)): # 총 날짜 개수
        tmp += lstH[j][i]
    avgCntStart.append((tmp / second) - stdList[i])
    avgCntEnd.append((tmp / second) + stdList[i])

#print(avgCntStart)
#print(avgCntEnd)
# -------------표준편차를 기준으로 탐색할 범위 지정 끝----------------

'''
탐색범위를 돌며, 영역 중 떨어지는 애를 구한다. 한 프레임 내에 10~50% 정도가 떨어지는 애들이면, 개 선택
'''

acceptFrameList = []
# acceptFrameList = [6.5, 7.0, 13.0, 13.5, 14.5, 15.0, 15.5, 16.0, 16.5, 20.5, 21.0, 21.5, 23.5, 52.5, 54.5, 57.5, 58.0, 58.5, 62.5, 65.0, 72.0, 73.5, 74.0, 74.5, 79.5, 80.0, 81.0, 82.5, 86.5, 92.0, 94.0, 103.0, 103.5, 108.0, 108.5, 109.0, 111.0, 111.5, 113.0, 115.5, 118.5, 119.0]
# 탐색범위 내의 프레임만 필터링

# lstH는, 한 화면의 각 격자의 평균 h값을 지님. 즉, 한 배열에 16개의 격자에 대한 평균h값.
for i in range(len(lstH)): # 총 날짜 개수
    tmparr=[]
    for j in range(len(lstH[i])): # 총 격자 개수
        if (lstH[i][j] < avgCntStart[j] and lstH[i][j] > avgCntStart[j] - stdList[j]) \
        or (lstH[i][j] > avgCntEnd[j] and lstH[i][j] < avgCntEnd[j] + stdList[j]) :
            tmparr.append(j)

    # 편차가 있는 곳이 1/3 이상 1/2 미만인 경우
    #if len(tmparr) > (heightcnt * widthcnt) // 3 \
    #        and len(tmparr) < (heightcnt * widthcnt) // 2:
    if len(tmparr) > (heightsplitter * widthsplitter) // 3 \
            and len(tmparr) < (heightsplitter * widthsplitter) // 2:
        acceptFrameList.append(i/2)
print('Accepted Frame is -', end = '\t')
print(acceptFrameList)
'''
영역 중 떨어지는 애를 구하는 과정 끝
'''

# --------------------보정과정. n초 이상 지속되는 경우에 하이라이트로 지정 -------------
highlightArr = []
startidx, endidx = 0, 0
frameidx = 0
consec = 0
for i in range(len(acceptFrameList)-1):
   if acceptFrameList[i+1] <= acceptFrameList[i]+1:
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

#highlightArr = [(38.5, 40.0), (116.0, 118.0), (120.0, 122.0), (133.0, 135.0), (137.0, 139.5), (142.5, 148.0), (244.0, 246.0), (340.5, 342.5), (344.0, 347.5), (374.0, 376.0), (385.0, 388.0), (468.5, 471.0), (692.0, 693.5), (798.5, 801.0)]
realhighlight = []
# 각 구간 별 차이가 5초 이하라면, 연결
for i in range(len(highlightArr)-1):
    if highlightArr[i+1][0] - highlightArr[i][1] < 4:
        realhighlight.append((highlightArr[i][0], highlightArr[i+1][1]))
        #del highlightArr[i+1], highlightArr[i]
        #highlightArr.append(newhighlight)
realhighlight.sort()
print(realhighlight)
#-------------------------------보정 끝-----------------------------#

# 이제, 롤 플레이 영상 한 3개만 다운받아서 확인해보자.
# 다음으로는 적절하게 퍼센티지 나눠보기

# # -----------------수집한 하이라이트 구간을 원본영상에서 자르기 ----------------------#
# # video.set(cv2.CV_CAP_POS_FRAMES, acceptFrameList[0])
parts = [(15, 30), (1200, 2400)]
cap = cv2.VideoCapture(filepath)
ret, frame = cap.read()
h, w, _ = frame.shape

# Define a fourcc (four-character code), and define a list of video writers;
# 3rd 파라미터: fps, 4th 파라미터: frame Size
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writers = [cv2.VideoWriter(f"{idx}.part{start}-{end}.avi", fourcc, fps, (w, h)) \
           for idx, (start, end) in enumerate(realhighlight)]

# Define a while loop, but before that, define a variable to
# keep track of which frame the while loop is at in the capture device:
f = 0
while ret: # ret == False시, 중지
    f += 1 # f == frame

    for i, part in enumerate(realhighlight):
        start, end = part
        if start <= f <= end:
            writers[i].write(frame)
    ret, frame = cap.read()

print(len(writers))
for writer in writers:
    writer.release()

cap.release()
# -----------------수집한 하이라이트 구간을 원본영상에서 자르기 끝 ----------------------#

# -----------------------동영상 합치기--------------------------#
from moviepy.editor import VideoFileClip, concatenate_videoclips

videofiles = [n for n in os.listdir('.') if n[-4:]=='.avi']
videofiles.sort()
print(videofiles)

final_clip = VideoFileClip(videofiles[0])
for i in range(len(videofiles)-1):
    final_clip = concatenate_videoclips([final_clip, VideoFileClip(videofiles[i+1])])
final_clip.write_videofile("finale.mp4")

print("time :", time.time() - starttime)  # 현재시각 - 시작시간 = 실행 시간

# ----------------------동영상 합치기 끝--------------------------#


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
