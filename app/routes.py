
from flask import Flask, render_template, Response, request
from app import app
from app import faceDetector as detector
import base64
import numpy as np
import cv2
import traceback

facebio = detector.FaceBio()

@app.route('/')
def index():
    return render_template('index.html')

# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen(cam.VideoCamera()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        init = True
        file = request.files['image']

        # Save file
        #filename = 'static/' + file.filename
        #file.save(filename)

        # Read image
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Detect faces
        faces, label = facebio.detect_faces(image)

        faceMaskDetected = label

        if len(faces) == 0:
            faceDetected = False
            num_faces = 0
            to_send = ''
        else:
            
            faceDetected = True
            num_faces = len(faces)
            
            # Draw a rectangle
            for item in faces:
                facebio.draw_frame(image, item['rect'])
            
           
        image_small = cv2.resize(image, (0,0), fx=0.3, fy=0.3) 
        image_content = cv2.imencode('.jpg', image_small)[1].tostring()
        encoded_image = base64.encodestring(image_content)
        to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    except Exception:
        #traceback.print_exc()
        init = False
        faceDetected = False
        num_faces = 0
        to_send = ''
        faceMaskDetected = False

    #print(faceDetected, faceMaskDetected, num_faces, init)
    return render_template('index.html', faceDetected=faceDetected, faceMaskDetected=faceMaskDetected, num_faces=num_faces, image_to_show=to_send, init=init)




