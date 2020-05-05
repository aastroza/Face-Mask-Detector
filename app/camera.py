import cv2
from fastai.vision import *
import torch

face_cascade=cv2.CascadeClassifier("./app/model/haarcascade_frontalface_default.xml")
ds_factor=0.75
face_mask_learn = load_learner('./app/model')

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        t = torch.tensor(np.ascontiguousarray(np.flip(image, 2)).transpose(2,0,1)).float()/255
        img = Image(t) # fastai.vision.Image, not PIL.Image
        pred_class, pred_idx, outputs = face_mask_learn.predict(img)
        
        if(str(pred_class) == 'mask'):
            color = (0, 255, 0)
            label = 'Con mascarilla'
        else:
            color = (0, 0, 255)
            label = 'Sin mascarilla'
        

        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            break
        
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
