import cv2 

#We need two classifiers one to detect frontal face and one to detect eye
face_clf=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile=cv2.CascadeClassifier('haarcascade_smile.xml')
# we define a detect function which takes a gray scale image to processs and a normal 
# image to return after processing

def facedetect(gray,frame):
    #for face we'll use the face classifier and a function which takes three
    # arguments - image,scaling factor (by how much it scales image) and min neighbours to check
    face=face_clf.detectMultiScale(gray,1.3,5)
    # face will return a tuple of 4 things- x coordinate ,y coord
    # width ,length of detected face
    
    #we will iterate through the faces to draw rectange over detected images
    
    for (x,y,w,h) in face:
        #we use rectangle function to draw rectangle on detected faces 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        # to detect smiles we will scan inside the faces and to do that we need
        # two regions of interest(roi) 1-for grascale 2- for original image
        
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        
        # just like we did for face we do for smiles
        smiles=smile.detectMultiScale(roi_gray,1.1,100)
        
        for (x,y,w,h) in smiles:
            cv2.rectangle(roi_color,(x,y),(x+w,y+h),(0,255,0),2)
            
    return frame

#Now we need to initialize webcam to record video
video_cap=cv2.VideoCapture(0)# 0 for internal webcam,1 for external webcam

while True:# We repeat infinitely (until break):
    _,frame=video_cap.read() # We get the last frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
    
    canvas = facedetect(gray, frame) # We get the output of our detect function.
    
    cv2.imshow('Video', canvas) # We display the outputs.
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break    
            
        
video_cap.release() # We turn the webcam off.
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.
        
    
    
    
