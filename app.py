import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier("cascade.xml")
cap = cv2.VideoCapture("stest.mp4")
itt = 0

while(cap.isOpened()):
    itt+=1
    ret, frame = cap.read()
    if itt % 10 == 0:
            
        if ret:
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 15)
            
            for (x, y, w, h) in faces:
                prediction = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                if prediction["emotion"]["neutral"] < 5:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_COMPLEX            
                    cv2.putText(frame, prediction["dominant_emotion"], (150, 150), font, 2, (0, 255, 0), 2, cv2.LINE_4)
            
            ims = cv2.resize(frame, (960, 960))

            cv2.imshow('Frame', ims)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break



cap.release()
cv2.destroyAllWindows()