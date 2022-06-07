import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

filepath = 'C:/Users/is7se/Desktop/코드용/videoCV/data_2/Lol_3.mp4'
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

gofps = 0
if fps >=30 :
    gofps = 30
else:
    gofps = fps

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
heightsplitter, widthsplitter = 3, 4 #3, 4 # 분할 개수.

# 0.5초 별 체크할 이미지의 사이즈
heightsz, widthsz = height//2, (width*3)//5

# 각 격자의 너비/높이별 개수
heightcnt = (int)(((27*height//40) - heightidx) / (height//heightsplitter))
widthcnt = (int)(((5*width//6) - widthidx) / (width//widthsplitter))

print(widthidx, widthmax, heightsz, heightsz//heightsplitter, widthcnt)
print(heightidx, heightmax, widthsz, widthsz//widthsplitter, heightcnt)
second = 0

#heightcnt = (int)(((2*height//3) - heightidx) / (height//10))
#widthcnt = (int)(((2*width//3) - widthidx) / (width//12))
# (640, 800, 960, 1120) - width / (180, 288, 396, 504, 612) - height
# 640 ~ 1280 & 180 ~ 720
# 160 = width(1920)/12, 108 = height(1080)/10

prevscreen, nowscreen = [], []

#heightsplitter, widthsplitter = 10, 12
lstH = []

# --------------------------- 그래프 용-------------------------
graphf, graphaxes = plt.subplots(heightsplitter, widthsplitter)
graphf.set_size_inches((20, 15))
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
graphf.suptitle('Subplot Example', fontsize = 15)
# plt그리기용
x_val = []
y_val = [[]for i in range(heightsplitter* widthsplitter)]
z_val = [[]for i in range(heightsplitter* widthsplitter)]
z_val2 = [[]for i in range(heightsplitter* widthsplitter)]

y_val_total = []
# --------------------------- 그래프 용 끝 -------------------------

while(video.isOpened()):
    ret, image = video.read()
    if image is None:
        break

    # 30 : 앞서 불러온 fps 값을 사용하여 0.5초마다 추출
    if(int(video.get(1)) % (gofps) == 0):
        if second%120 == 0: #60초에 한 번
            print(second, end = ' ')
        averH = 0 # 합산시킬 H
        totalH = 0
        x_val.append(second)

        lstfornowscreen = [] #같은 프레임 내의 이미지
        graphyidx = 0 # 그래프에 넣기 용
        # (256 448 640 832 1024) (126 216 306 396 486)
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
                                H = ((((Max - newB)/6) + (del_max/2)) / del_max) \
                                     - ((((Max - newG)/6)+(del_max/2))/del_max)
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

                y_val[graphyidx].append(averH) #그래프용에 averH 삽입
                graphyidx += 1

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
        #if second == 600:
        #    break

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

    for k in range(len(lstH)):
        z_val[i].append((tmp / second) - stdList[i])
        z_val2[i].append((tmp / second) + stdList[i])


#print(avgCntStart)
#print(avgCntEnd)


#plt.show()
# -------------표준편차를 기준으로 탐색할 범위 지정 끝----------------

'''
탐색범위를 돌며, 영역 중 떨어지는 애를 구한다. 한 프레임 내에 10~50% 정도가 떨어지는 애들이면, 개 선택
'''

acceptFrameList = []

# lstH는, 한 화면의 각 격자의 평균 h값을 지님. 즉, 한 배열에 16개의 격자에 대한 평균h값.
for i in range(len(lstH)): # 총 날짜 개수
    tmparr=[]
    for j in range(len(lstH[i])): # 총 격자 개수
        if (lstH[i][j] < avgCntStart[j] and lstH[i][j] > avgCntStart[j] - stdList[j]) or \
        (lstH[i][j] > avgCntEnd[j] and lstH[i][j] < avgCntEnd[j] + stdList[j]) :
            tmparr.append(j)

    # 편차가 있는 곳이 1/6 이상 3/5 미만인 경우[2~9 -> 5~12]
    #if len(tmparr) > (heightcnt * widthcnt) // 3 \
    #        and len(tmparr) < (heightcnt * widthcnt) // 2:
    if len(tmparr) > (heightsplitter * widthsplitter) // 5 \
            and len(tmparr) < (heightsplitter * widthsplitter)*3 // 5:
        acceptFrameList.append(i)
print('Accepted Frame is -', end = '\t')
print(acceptFrameList)
'''
영역 중 떨어지는 애를 구하는 과정 끝
'''

# ------------------- 영역을 안나누는 경우 -----------------------#
# video = cv2.VideoCapture(filepath)
# second = 0
# while(video.isOpened()):
#     ret, image = video.read()
#     if image is None:
#         break
#
#     # 앞서 불러온 fps 값을 사용하여 0.5초마다 추출
#     if(int(video.get(1)) % (gofps) == 0):
#         x_val.append(second)
#         if second%120 == 0: #60초에 한 번
#             print(second, end = ' ')
#         averH = 0 # 합산시킬 H
#         for i in range(heightidx, heightmax):
#         #for i in range(0, height):
#             tmparr = []
#             for j in range(widthidx, widthmax):
#             #for j in range(0, width):
#                 newR, newG, newB = (image[i][j][0]/255),(image[i][j][1]/255),(image[i][j][2]/255)
#                 Min = min(newR, newG, newB)
#                 Max = max(newR, newG, newB)
#                 del_max = Max - Min
#                 H = 0
#
#                 #H값 추출
#                 if(del_max == 0):
#                     H = 0
#                 else:
#                     if newR == Max:
#                         H = ((((Max - newB)/6) + del_max/2)) / ((((Max - newG)/6)+(del_max/2))/del_max)
#                     elif newG == Max:
#                         H = (1/3)+((((Max-newR)/6)+(del_max/2)) / del_max) - ((((Max - newB)/6)+(del_max/2))/del_max)
#                     elif newB == Max:
#                         H = (2/3)+((((Max-newG)/6)+(del_max/2)) / del_max) - ((((Max - newR)/6)+(del_max/2))/del_max)
#                     if H<0:
#                         H += 1
#                     if H>1:
#                         H -= 1
#                 #tmparr.append((H, 0, 1))
#                 averH += H
#             #prevscreen.append(tmparr)
#         #print(averH)
#         #x_val.append(averH)
#         averH /= (widthsz * heightsz) # 한 프레임의 H값 평균
#         lstH.append(averH) # 이걸 리스트에 넣기
#
#         second += 1
#     #if second == 80:
#     #    break
# video.release()
#
# std = np.std(lstH) # 표준편차
# print(std)
#
# # 표준편차를 기준으로 탐색할 범위 지정. H의 평균값 +- 표준편차가 일반구간이 됨.
# avgCntStart = (sum(lstH) / len(lstH)) - std  # 일반구간의 시작기준.
# avgCntEnd = (sum(lstH) / len(lstH)) + std # 일반구간의 끝기준.
# acceptFrameList = []
# z_val_total, z_val2_total = [], []
# for i in range(len(lstH)):
#     y_val_total.append(lstH[i])
#     z_val_total.append(avgCntStart)
#     z_val2_total.append(avgCntEnd)
#
# # print(std, avgCntStart, avgCntEnd)
#
# # 탐색범위 내의 프레임만 필터링
# for i in range(len(lstH)):
#     if (lstH[i] < avgCntStart and lstH[i] > avgCntStart - std) \
#         or (lstH[i] > avgCntEnd and lstH[i] < avgCntEnd + std) :
#             acceptFrameList.append(i)
# print(acceptFrameList)
# plt.plot(x_val, y_val_total)
# plt.plot(x_val, z_val_total)
# plt.plot(x_val, z_val2_total)
#
# plt.vlines(965, 0, 1, color='gray', linestyle='solid', linewidth=2)
# plt.vlines(970, 0, 1, color='gray', linestyle='solid', linewidth=2)
# plt.vlines(740, 0, 1, color='gray', linestyle='solid', linewidth=2)
# plt.vlines(750, 0, 1, color='gray', linestyle='solid', linewidth=2)
# plt.vlines(1213, 0, 1, color='gray', linestyle='solid', linewidth=2)
# plt.vlines(1223, 0, 1, color='gray', linestyle='solid', linewidth=2)
#
# #plt.show()
# plt.savefig('lol3_nogrid_allvar_frontback.png')
# ------------------- 영역을 안나누는 경우 끝-----------------------#

# --------------------보정과정. n초 이상 지속되는 경우에 하이라이트로 지정 -------------
highlightArr = []
startidx, endidx = 0, 0
frameidx = 0
consec = 0
for i in range(len(acceptFrameList)-1):
   if acceptFrameList[i+1] <= acceptFrameList[i]+1:
   #if acceptFrameList[i+1] <= acceptFrameList[i]+2:
       # 만약 연속이 시작될 경우(그전까지 0이었다면)
       if consec == 0:
           startidx = acceptFrameList[i] # 시작 idx 설정
       consec += 1 # 연속 = 1 증가

       if i == len(acceptFrameList)-2 and consec >= 3 and acceptFrameList[i]+1 == acceptFrameList[i+1]:
            endidx = acceptFrameList[i]+1
            highlightArr.append([startidx, endidx])
            consec, startidx, endidx = 0, 0, 0
   else :
       if consec >= 3: # 4번 이상 충분히 유지되었다면, endidx를 설정
           endidx = acceptFrameList[i]
           highlightArr.append([startidx, endidx])
       consec, startidx, endidx = 0, 0, 0

# highlightArr = [(38.5, 40.0), (116.0, 118.0), (120.0, 122.0), (133.0, 135.0), (137.0, 139.5), (142.5, 148.0), (244.0, 246.0), (340.5, 342.5), (344.0, 347.5), (374.0, 376.0), (385.0, 388.0), (468.5, 471.0), (692.0, 693.5), (798.5, 801.0)]

#--------------------------------추가보정 = 2초씩 붙이고, 각 구간별 좌우 판별, 2초 차이나면 붙이기 ------------------#
realhighlight = []
# 양옆에 2초씩 더함.
for i in range(len(highlightArr)):
    if highlightArr[i][0] > 2:
        highlightArr[i][0] -= 1
    if highlightArr[i][1] < ((length//fps))-3:
        highlightArr[i][1] += 3

i = 0
while i < len(highlightArr)-1 :
    if i >= len(highlightArr):
        break
    if highlightArr[i+1][0] - highlightArr[i][1] <= 2 or highlightArr[i+1][0] < highlightArr[i][1]:
        highlightArr[i][1] = highlightArr[i+1][1]
        highlightArr.pop(i+1)
        # del highlightArr[i+1]
    else:
        i += 1

# -------------------------------보정 끝-----------------------------#

# 이제, 롤 플레이 영상 한 3개만 다운받아서 확인해보자.
# 다음으로는 적절하게 퍼센티지 나눠보기

# -----------------수집한 하이라이트 구간을 원본영상에서 자르기 ----------------------#
# # video.set(cv2.CV_CAP_POS_FRAMES, acceptFrameList[0])

# 프레임별로 잘려있으니, 이를 보기 편하게 프레임만큼 곱해 초로 볼 수 있게.
for i in range(len(highlightArr)):
     highlightArr[i][0] *= fps
     highlightArr[i][1] *= fps
print(highlightArr)

#for i in range(len(realhighlight)):
#    realhighlight[i][0] *= fps
#    realhighlight[i][1] *= fps
#print(realhighlight)

cap = cv2.VideoCapture(filepath)
ret, frame = cap.read()
h, w, _ = frame.shape

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writers = [cv2.VideoWriter(f"{idx}_{start//gofps}_{end//gofps}.avi", fourcc, fps, (w, h)) \
           for idx, (start, end) in enumerate(highlightArr)]
#writers = [cv2.VideoWriter(f"{idx}_{start//gofps}_{end//gofps}.avi", fourcc, fps, (w, h)) \
#           for idx, (start, end) in enumerate(realhighlight)]
f = 0
while ret: # ret == False시, 중지
    f += 1 # f == frame. 초당 30 * 60초 = 1800

    #for i, part in enumerate(realhighlight):
    for i, part in enumerate(highlightArr):
        start, end = part
        if start <= f <= end:
            writers[i].write(frame)
    ret, frame = cap.read()
    if f%(fps*60) == 0: #60초마다
        print(f/30, end=' ')
    if frame is None:
         break

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
print(len(videofiles))
final_clip = VideoFileClip(videofiles[0])
for i in range(len(videofiles)-1):
    #print(i+1, end=' ')
    print(videofiles[i+1])
    final_clip = concatenate_videoclips([final_clip, VideoFileClip(videofiles[i+1])])
final_clip.write_videofile("finale.mp4")

print("time :", time.time() - starttime)  # 현재시각 - 시작시간 = 실행 시간

# ----------------------동영상 합치기 끝--------------------------#

# 그래프 저장
for i in range(heightsplitter):
    for j in range(widthsplitter):
        graphaxes[i,j].plot(x_val, y_val[i*heightsplitter+j])
        graphaxes[i,j].plot(x_val, z_val[i*heightsplitter+j])
        graphaxes[i,j].plot(x_val, z_val2[i*heightsplitter+j])
plt.savefig('lol3_nogrid_allvar_frontback.png')
