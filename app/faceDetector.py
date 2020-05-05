import cv2
from fastai.vision import *
import torch


#face_cascade=cv2.CascadeClassifier("./app/model/haarcascade_frontalface_default.xml")
#ds_factor=0.75
#face_mask_learn = load_learner('./app/model')
class FaceBio():
    def __init__(self):
        # self.device = torch.device('cpu')

        # self.mtcnn = MTCNN(
        #     image_size=160, margin=0, min_face_size=20,
        #     thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        #     device=self.device
        # )

        #self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.face_cascade=cv2.CascadeClassifier("./app/model/haarcascade_frontalface_default.xml")
        self.face_mask_learn = load_learner('./app/model')


    def detect_faces(self, image):
        '''Detect face in an image'''
        
        faces_list = []

        t = torch.tensor(np.ascontiguousarray(np.flip(image, 2)).transpose(2,0,1)).float()/255
        img = Image(t) # fastai.vision.Image, not PIL.Image
        pred_class, pred_idx, outputs = self.face_mask_learn.predict(img)
        
        if(str(pred_class) == 'mask'):
            color = (0, 255, 0)
            label = True
        else:
            color = (0, 0, 255)
            label = False

        #faces, probs = self.mtcnn.detect(image)

        # Convert the test image to gray scale (opencv face detector expects gray images)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect multiscale images (some images may be closer to camera than others)
        # result is a list of faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

        # If not face detected, return empty list  
        if  len(faces) == 0:
            #face_dict['label'] = label
            #faces_list.append(face_dict)
            return faces_list, label
        else:
            for i in range(0, len(faces)):
                (x, y, w, h) = faces[i]
                #print(x, y, w, h)
                face_dict = {}
                face_dict['face'] = gray[int(y):int(y) + int(w), int(x):int(x) + int(h)]
                face_dict['rect'] = (x, y, w, h)
                #face_dict['label'] = label
                #face_dict['color'] = color
                faces_list.append(face_dict)

        # Return the face image area and the face rectangle
        return faces_list, label

    def draw_frame(self, image, rect):
        (x, y, w, h) = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)      
        #cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

