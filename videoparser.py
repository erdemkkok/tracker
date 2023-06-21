import cv2
import numpy as np

cap = cv2.VideoCapture(0)
count=0
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(200,200))
    box = cv2.selectROI("frame", frame)
    x,y,w,h=box
    img=frame[y:y+h,x:x+w]
    a=[x/200,(x+w)/200,y/200,(y+h)/200]
    print(x/200,(x+w)/200,y/200,(y+h)/200)
    with open('readme.txt', 'a') as f:
        f.write(str(a))
        f.write('\n')

    #frame=frame[50:70,50:70]
    cv2.imwrite("areas/{}.jpeg".format(count),img)
    count+=1
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)
    if key== ord('t'):
        box = cv2.selectROI("frame", frame)
        x,y,w,h=box
        #frame=frame[y:y+h,x:x+w]
        print(x/200,(x+w)/200,y/200,(y+h)/200)
        #cv2.imwrite("areas/{}.jpeg".format(count),frame)


    if key == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()


