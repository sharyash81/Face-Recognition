from matplotlib import pyplot as plt
import cv2
import numpy as np
import copy

!gdown --id 13f_vZ51V4hUO-kuvXnqlb4Nq6fgdxHZO &> /dev/null
!gdown --id 1AaFNIVGPG8SkcmKduvyMOrwuEJWIyVtt &> /dev/null
!python setup.py

from utils import *

#Array for detected and embeded frames
Frames = []
#Array for all 10x frames
Framess = []
#Array for plans of video
Plans = [ [] ]

planCompare = [0]
frameCounter = 1
pathOfVid = '/content/drive/MyDrive/Friends/f720_1280.mp4'
cap = cv2.VideoCapture(pathOfVid)
frameNum = int (cap.get(cv2.CAP_PROP_FRAME_COUNT))


color=[(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),
       (128,64,0),(64,128,0),(0,64,128),(0,128,64),(64,0,128),(128,0,64),(220,64,64),
       (64,220,64),(64,64,220),(220,128,220),(150,20,110),(125,250,64),(55,55,55),
       (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
       (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
       (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
       (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
       (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
       (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
       (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
       (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
       (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
       (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
       (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
       (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
       (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
       (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
       (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]
name=["Target1","Target2","Target3","Target4","Target5","Target6","Target7",
      "Target8","Target9","Target10","Target11","Target12","Target13","Target14",
      "Target15","Target16","Target17","Target18","Target19","Target20","Target21",
      "Target","Target","Target","Target","Target","Target","Target","Target",
      "Target","Target","Target","Target","Target","Target","Target","Target",
      "Target","Target","Target","Target","Target","Target","Target","Target",
      "Target","Target","Target","Target","Target","Target","Target","Target",
      "Target","Target","Target","Target","Target","Target","Target","Target",
      "Target","Target","Target","Target","Target","Target","Target","Target",
      "Target","Target","Target","Target","Target","Target","Target","Target",
      "Target","Target","Target","Target","Target","Target","Target","Target",
      "Target","Target","Target","Target","Target","Target","Target","Target",
      "Target","Target","Target","Target","Target","Target","Target","Target",
      "Target","Target","Target","Target","Target","Target","Target","Target",
      "Target","Target","Target","Target","Target","Target","Target","Target",
      "Target","Target","Target","Target","Target","Target","Target","Target",
      "Target","Target","Target","Target","Target","Target","Target","Target",
      "Target","Target","Target","Target","Target","Target","Target","Target",
      "Target","Target","Target","Target","Target","Target","Target","Target"]
vidname=["vid1.mp4","vid2.mp4","vid3.mp4","vid4.mp4","vid5.mp4","vid6.mp4","vid7.mp4",
         "vid8.mp4","vid9.mp4","vid10.mp4","vid11.mp4","vid12.mp4","vid13.mp4","vid14.mp4",
         "vid15.mp4","vid16.mp4","vid17.mp4","vid18.mp4","vid19.mp4","vid20.mp4","vid21.mp4",
         "vid22.mp4","vid23.mp4","vid24.mp4","vid25.mp4","vid26.mp4","vid27.mp4","vid28.mp4",
         "vid29.mp4","vid30.mp4","vid31.mp4","vid32.mp4","vid33.mp4","vid34.mp4","vid35.mp4",
         "vid36.mp4","vid37.mp4","vid38.mp4","vid39.mp4","vid40.mp4","vid41.mp4","vid42.mp4",
         "vid43.mp4","vid44.mp4","vid45.mp4","vid46.mp4","vid47.mp4","vid48.mp4","vid49.mp4",
         "vid50.mp4","vid51.mp4","vid52.mp4","vid53.mp4","vid54.mp4","vid55.mp4","vid56.mp4",
         "vid57.mp4","vid58.mp4","vid59.mp4","vid60.mp4","vid61.mp4","vid62.mp4","vid63.mp4",
         "vid64.mp4","vid65.mp4","vid66.mp4","vid67.mp4","vid68.mp4","vid69.mp4","vid70.mp4",
         "vid71.mp4","vid72.mp4","vid73.mp4","vid74.mp4","vid75.mp4","vid76.mp4","vid77.mp4",
         "vid78.mp4","vid79.mp4","vid80.mp4","vid81.mp4","vid82.mp4","vid83.mp4","vid84.mp4",
         "vid85.mp4","vid86.mp4","vid87.mp4","vid88.mp4","vid89.mp4","vid90.mp4","vid91.mp4",
         "vid92.mp4","vid93.mp4","vid94.mp4","vid95.mp4","vid96.mp4","vid97.mp4","vid98.mp4",
         "vid99.mp4","vid100.mp4","vid101.mp4","vid102.mp4","vid103.mp4","vid104.mp4",
         "vid105.mp4","vid106.mp4","vid107.mp4","vid108.mp4","vid109.mp4","vid110.mp4"]

#Start
while (cap.isOpened() and frameCounter <frameNum-20):
  cap.set(cv2.CAP_PROP_POS_FRAMES, frameCounter)
  ret, frame = cap.read()
  if ret == False:
    break
  bboxesf, lmarksf, cropsf, embeddingsf = detect_and_embed(frame)

  if (frameCounter<10):
    #Set Scale
    heightVid=frame.shape[0]
    widthVid=frame.shape[1]
    diff=int((127)*(heightVid/1080))
    #Unusual
    if (heightVid>1200):
      s1=0.5
      s2=0.3
      m1=0.3
      w1=0.3
      w2=0.3
      comp=90000
      thickness=4
      scale=1.4
    #1080
    elif (heightVid>900):
      s1=0.5
      s2=0.3
      m1=0.3
      w1=0.3
      w2=0.3
      comp=65000
      thickness=3
      scale=1.1
    #720
    elif (heightVid>600):
      s1=0.5
      s2=0.3
      m1=0.3
      w1=0.3
      w2=0.25
      comp=15000
      thickness=2
      scale=0.9
    #480
    elif (heightVid>420):
      s1=0.5
      s2=0.3
      m1=0.3
      w1=0.3
      w2=0.3
      comp=7500
      thickness=2
      scale=0.7
    #360
    elif (heightVid>240):
      s1=0.5
      s2=0.3
      m1=0.3
      w1=0.3
      w2=0.25
      comp=5000
      thickness=1
      scale=0.5
    #Unusual
    else:
      s1=0.5
      s2=0.25
      m1=0.25
      w1=0.25
      w2=0.25
      comp=4000
      thickness=1
      scale=0.3
  
  #Part 1: Compare plans
  Plans[-1].append(frame)

  #Frame 1
  image = frame.copy()
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  histogram = cv2.calcHist([gray_image], [0],None, [256], [0, 256])
  cap.set(cv2.CAP_PROP_POS_FRAMES, frameCounter+10)
  ret, frame = cap.read()
  if ret == False:
    break
  #Frame 2
  image1 = frame.copy() 
  gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 
  histogram1 = cv2.calcHist([gray_image1], [0],None, [256], [0, 256]) 
  planCompare = 0

  #Compare
  i = 0
  while i<len(histogram) and i<len(histogram1): 
      planCompare+=(histogram[i]-histogram1[i])**2
      i += 1
  planCompare = planCompare**(1 / 2)
  
  if planCompare[0]>comp:
    #A new plan begins
    Plans.append([])

  #Add frames
  cap.set(cv2.CAP_PROP_POS_FRAMES, frameCounter)
  ret, frame = cap.read()
  if ret == False:
    break
  bboxesf, lmarksf, cropsf, embeddingsf = detect_and_embed(frame)
  Framess.append(frame)
  Frames.append([bboxesf,lmarksf,cropsf,embeddingsf])

  frameCounter+=10
  cap.set(cv2.CAP_PROP_POS_FRAMES, frameCounter)
cap.release()

#Part 2: Find the characters of each plan
#Array for characters
Chars=[]
#Array for properties of characters(Frame number and character number in frame)
Charst=[]

w = -1

#Find and separate characters with two Method(Location and Similarity)
for t in range(len(Plans)):
  Chars.append([])
  Charst.append([])
  for i in range(len(Plans[t])):
    w+=1
    for j in range(len(Frames[w][0])):
      flag=0
      for k in range(len(Chars[t])):
        flagak=0
        for l in range(len(Chars[t])):
          if (Chars[t][l][0][0]-diff<Frames[w][0][j][0] and
              Chars[t][l][0][0]+diff>Frames[w][0][j][0] and
              Chars[t][l][0][1]-diff<Frames[w][0][j][1] and
              Chars[t][l][0][1]+diff>Frames[w][0][j][1]):
            flagak+=1
            if flagak==2:
              break
        if flagak<2:
          if (cosine_similarity(Chars[t][k][3], Frames[w][3][j])[0,0]>s1 or
             (Chars[t][k][0][0]-diff<Frames[w][0][j][0] and
              Chars[t][k][0][0]+diff>Frames[w][0][j][0] and
              Chars[t][k][0][1]-diff<Frames[w][0][j][1] and
              Chars[t][k][0][1]+diff>Frames[w][0][j][1])):
            flag=1
            Chars[t][k][0]=Frames[w][0][j].copy()
            Chars[t][k][1]=Frames[w][1][j].copy()
            Chars[t][k][2]=Frames[w][2][j].copy()
            Chars[t][k][3]=Frames[w][3][j].copy()
            Charst[t][k].append([w,j])
            break
        else:
          if (cosine_similarity(Chars[t][k][3], Frames[w][3][j])[0,0]>s2):
            flag=1
            Chars[t][k][0]=Frames[w][0][j].copy()
            Chars[t][k][1]=Frames[w][1][j].copy()
            Chars[t][k][2]=Frames[w][2][j].copy()
            Chars[t][k][3]=Frames[w][3][j].copy()
            Charst[t][k].append([w,j])
            break
      if flag==0:
        Chars[t].append([Frames[w][0][j],Frames[w][1][j],Frames[w][2][j],Frames[w][3][j]])
        Charst[t].append([[w,j]])

#Part 3: Connecting plans together and find common characters
for t in range(len(Charst)-1):
  #Array for Marking removed characters
  Charsel=[]
  for i in range(len(Charst[0])):
    if (len(Charst[0][i])>=7):
      numi=7
    else:
      numi=len(Charst[0][i])
    for j in range(len(Charst[t+1])):
      if (len(Charst[t+1][j])>=7):
        numj=7
      else:
        numj=len(Charst[t+1][j])
      sum=0
      for k in range(numi):
        for l in range(numj):
          sum+=cosine_similarity(Frames[Charst[0][i][k*(len(Charst[0][i])//numi)][0]][3][Charst[0][i][k*(len(Charst[0][i])//numi)][1]],
                                 Frames[Charst[t+1][j][l*(len(Charst[t+1][j])//numj)][0]][3][Charst[t+1][j][l*(len(Charst[t+1][j])//numj)][1]])[0,0]
      sum/=(numi*numj)
      if sum>m1:
        if j not in Charsel:
          Charst[0][i]+=Charst[t+1][j]
          Charsel.append(j)
          break
  for i in range(len(Charst[t+1])):
    if i not in Charsel:
      Charst[0].append(Charst[t+1][i])

#Part 4: Reassemble similar characters
Charsel=[]
for i in range(len(Charst[0])):
  for j in range(len(Charst[0])-1):
    if(len(Charst[0][j])>len(Charst[0][j+1])):
      t=copy.deepcopy(Charst[0][j])
      Charst[0][j]=copy.deepcopy(Charst[0][j+1])
      Charst[0][j+1]=copy.deepcopy(t)
for i in range(len(Charst[0])):
  if (len(Charst[0][i])>=10):
    numi=10
  else:
    numi=len(Charst[0][i])
  for j in range(i+1,len(Charst[0])):
    w=0
    if (len(Charst[0][j])>=10):
      numj=10
    else:
      numj=len(Charst[0][j])
    for k in range(numi):
      for l in range(numj):
        if cosine_similarity(Frames[Charst[0][i][k*(len(Charst[0][i])//numi)][0]][3][Charst[0][i][k*(len(Charst[0][i])//numi)][1]],
          Frames[Charst[0][j][l*(len(Charst[0][j])//numj)][0]][3][Charst[0][j][l*(len(Charst[0][j])//numj)][1]])[0,0]>w1:
          w+=1
    if (w/(numi*numj)>w2):
      Charsel.append(i)
      Charst[0][j]+=Charst[0][i]
      break

#Part 5: Remove fake characters
NewChar=[]
w=0
for i in range(len(Charst[0])):
  if i not in Charsel:
    NewChar.append(Charst[0][i])
for i in range(len(NewChar)):
  if (len(NewChar[w])<(frameNum)/600):
    NewChar.pop(w)
    w-=1
  w+=1
for i in range(len(NewChar)):
  NewChar[i].sort()

#Part 6: Calculate the time of presence of the character
#Array for show the total time of the character presence
Time=[]
#Array for show the presence of a character in the film so far
Timer=[]
for i in range(len(NewChar)):
  Time.append(len(NewChar[i]))
  Timer.append([0,0])

#Part 7: Creating videos of the presence of characters
cap= cv2.VideoCapture(pathOfVid)
for i in range(len(NewChar)):
  out = cv2.VideoWriter(vidname[i],cv2.VideoWriter_fourcc(*'DIVX'), 30, (widthVid,heightVid))
  for j in range(len(NewChar[i])):
    f=(NewChar[i][j][0]*10)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    for k in range(10):
      ret, frame = cap.read()
      if ret == False:
        break
      cv2.rectangle(frame, (Frames[NewChar[i][j][0]][0][NewChar[i][j][1]][0], Frames[NewChar[i][j][0]][0][NewChar[i][j][1]][1]),
                   (Frames[NewChar[i][j][0]][0][NewChar[i][j][1]][2], Frames[NewChar[i][j][0]][0][NewChar[i][j][1]][3]), color[i], thickness)
      cv2.putText(frame, name[i] +" "+str(Timer[i][0])+"."+ str (Timer[i][1]),
                  (Frames[NewChar[i][j][0]][0][NewChar[i][j][1]][0], int (Frames[NewChar[i][j][0]][0][NewChar[i][j][1]][1]-12)),
                   cv2.FONT_HERSHEY_SIMPLEX,scale , color[i], thickness, cv2.LINE_AA)
      out.write(frame)
      Timer[i][1]+=1
      if Timer[i][1]==30:
        Timer[i][0]+=1
        Timer[i][1]=0
  out.release()
cap.release()
cv2.destroyAllWindows
