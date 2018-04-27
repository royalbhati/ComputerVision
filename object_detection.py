# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Defining a function that will do the detections
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame

# Creating the SSD neural network
net = build_ssd('test')
net.load_state_dict(torch.load('VGG_coco_SSD_300x300.h5', map_location = lambda storage, loc: storage))

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))



# video_cap=cv2.VideoCapture(0)

# while True:# We repeat infinitely (until break):
#     _,frame=video_cap.read() # We get the last frame.
#  # Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Defining a function that will do the detections
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame

# Creating the SSD neural network
net = build_ssd('test')
net.load_state_dict(torch.load('VGG_coco_SSD_300x300.h5', map_location = lambda storage, loc: storage))

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))



video_cap=cv2.VideoCapture(0)

while True:# We repeat infinitely (until break):
    _,frame=video_cap.read() # We get the last frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
    
     # We get the output of our detect function.
    frame = detect(frame, net.eval(), transform)
    cv2.imshow('Video', frame)
     # We display the outputs.
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break    
            
        
video_cap.release() # We turn the webcam off.
cv2.destroyAllWindows()




#For doing detection on a video

#Doing some Object Detection on a video
# reader = imageio.get_reader('epic_horses.mp4')
# fps = reader.get_meta_data()['fps']
# writer = imageio.get_writer('output.mp4', fps = fps)
# print(len(reader))
# for i, frame in enumerate(reader):
#     frame = detect(frame, net.eval(), transform)
#     writer.append_data(frame)
#     print(i)
writer.close()   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
    
#      # We get the output of our detect function.
#     frame = detect(frame, net.eval(), transform)
#     cv2.imshow('Video', frame)
#      # We display the outputs.
    
#     if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
#         break    
            
        
# video_cap.release() # We turn the webcam off.
# cv2.destroyAllWindows()




#For doing detection on a video

#Doing some Object Detection on a video
#reader = imageio.get_reader('epic_horses.mp4')
#fps = reader.get_meta_data()['fps']
#writer = imageio.get_writer('output.mp4', fps = fps)
#print(len(reader))
# for i, frame in enumerate(reader):
#     frame = detect(frame, net.eval(), transform)
#     writer.append_data(frame)
#     print(i)
writer.close()
