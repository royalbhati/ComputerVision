# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Defining a function that will do the detections
#it will work on frame by frame rather thn video at once
#we will take three arguments 1)frame(image) 2)net(SSD neural net) 3)tranform(to have compatible images as input)
def detect(frame,net,transform):
    #frame.shape will return a vector containing three values
    #height,width and no, of channels(0 for b/w and 3 for color)
    h,w=frame.shape[:2]
    # we need to make certian tranformatios on the frame so that
    #it is of right dimensions and colors as required by the neural network
    
    frame_t=transform(frame)[0]# we just need the first element
    #now we need to coonver the numpy array we got into tensor variable
    #to do that we need torch 
    
    #Also we neural network in torch is trained for green red blue but
    #we have red blue green so we need to transform
    x=torch.from_numpy(frame_t).permute(2,0,1)
    
    #since neural network accepts input as batches so we need to add fake dimension
    #we add zero to add first dimension
    #now we need to convert tensor var into torch variable
    x = Variable(x.unsqueeze(0))    
    
    #feeding input variable to neura net
    y=net(x)
    
    # to get data from output var y
    detections=y.data
    
    #to normalize the detections we need 4 dimensions to scale 
    #first hght,width would contain scalar values of upper left rectangle and second h,w
    #for scalar values of lower right corner
    
    scale=torch.Tensor([w,h,w,h])
    
    #detection wil cntain 
    #detections=[batch,no. of classes(for classifying varoius objects),no of
    #           occurences,(score,x0,x1,h0,y1) ]
    
    for i in range(detections.size(1)):# For every class:
        
        
     # We initialize the loop variable j that will correspond to the occurrences of the class.
        j=0
     
     # We take into account all the occurrences j of the class i that have a matching
     #score larger than 0.6.
        while detections[0, i, j, 0] >= 0.6: 
    #We get the coordinates of the points at the upper left and the lower right of the detector rectangle.
    #also wee need to normalize using scale and convert into numpy array because
    #draw rectangle func in opencv works using arrays
            pt=(detections[0,i,j,1:]*scale).numpy
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
writer.close()
    
    
    
    
    
    
    
