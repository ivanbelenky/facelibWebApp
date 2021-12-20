import os
import sys
import pickle



from flask import Flask, redirect, jsonify, request, url_for, render_template, flash
from flask.helpers import send_from_directory


app = Flask(__name__, static_folder='static', template_folder='templates')

app.debug = True
app.config["IMAGE_UPLOADS"] = sys.path[0]+"/uploads/"
app.config["DETECTION"] = sys.path[0]+"/processed/detection/"
app.config["AGE_GENDER_EMOTION"] = sys.path[0] + "/processed/age_gender_emotion/"


def delete_old_images(path):
    for file in os.listdir(path):
        os.remove(path+file)

def get_image(path):
    for file in os.listdir(path):
        return path+file

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/",methods=['GET','POST'])
def home():
    #delete_old_images(os.path.join(app.config["IMAGE_UPLOADS"]))
    if not os.listdir(os.path.join(app.config["IMAGE_UPLOADS"])):
        return render_template('index.html')  
    else:
        return render_template('index.html',
         uploaded_image=get_image(os.path.join(app.config["IMAGE_UPLOADS"])))


# Route to upload image

@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            if image.content_type == "application/octet-stream":
                delete_old_images(os.path.join(app.config["IMAGE_UPLOADS"]))
                delete_old_images(os.path.join(app.config["DETECTION"]))
                delete_old_images(os.path.join(app.config["AGE_GENDER_EMOTION"]))    
                return render_template("index.html")

            delete_old_images(os.path.join(app.config["IMAGE_UPLOADS"]))
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            return render_template("index.html", uploaded_image=image.filename)    
        

    return render_template("index_html")

@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    from flask import send_from_directory
    return send_from_directory(app.config["IMAGE_UPLOADS"], filename)


@app.route('/detection/<filename>')
def send_detection_file(filename=''):
    from flask import send_from_directory
    return send_from_directory(app.config["DETECTION"], filename)


@app.route('/A_G_E/<filename>')
def send_A_G_E_file(filename=''):
    from flask import send_from_directory
    return send_from_directory(app.config['AGE_GENDER_EMOTION'],filename)





from facelib import FaceDetector,EmotionDetector, AgeGenderEstimator
import cv2

from face_detect import create_face_segregation_images, create_face_landmarks_images
from a_g_e_detect import get_age_gender_emotion, create_images_age_gender_emotion


def get_image_tensor():
    img_path = get_image(os.path.join(app.config["IMAGE_UPLOADS"]))
    img = cv2.imread(img_path,-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img




def get_face_detector(model="default"):
    """
    Args:
        model (str, optional): [description]. Defaults to mobilenet.
        backbones available (resnet, mobilenet, slim, rfb)
    """

    with open(f"facelib_models/{model}_facedetector.modlib","rb") as f:
        detector = pickle.load(f)
        return detector


def get_age_gender_detector():
    with open("facelib_models/default_age_gender_detector.modlib","rb") as f:
        detector = pickle.load(f)
        return detector


def get_emotion_detector(model="default"):
    """
    Args:
        model (str, optional): [description]. Defaults to densenet121.
        backbones available (resnet34, densenet121)
    """

    with open(f"facelib_models/{model}_emotion_detector.modlib","rb") as f:
        detector = pickle.load(f)
        return detector



    
@app.route('/facedetect', methods=['GET'])
def send_face_detection_analysis():
    if not os.listdir(os.path.join(app.config["IMAGE_UPLOADS"])):
        return "No image to perform an analysis, please upload one"
    
    delete_old_images(os.path.join(app.config["DETECTION"])) 

    img = get_image_tensor()
    detector = get_face_detector()
    faces, boxes, scores, landmarkss = detector.detect_align(img)

    if faces.numel() == 0:
        return "No faces were found on this image, try with another one"

    create_face_segregation_images(faces)
    create_face_landmarks_images(img, faces, boxes,scores, landmarkss)

    return render_template("face_detect.html",
        face_segregation = 'face_segregation.svg',
        face_markers = 'face_markers.svg')




@app.route('/agegenderemotion',methods=['GET'])
def send_age_gender_emotion_analysis():
    if not os.listdir(os.path.join(app.config["IMAGE_UPLOADS"])):
        return "No image to perform an analysis, please upload one"
    
    delete_old_images(os.path.join(app.config["AGE_GENDER_EMOTION"])) 

    img = get_image_tensor()
    face_detector = get_face_detector()
    age_gender_detector = get_age_gender_detector()
    emotion_detector = get_emotion_detector()
    
    faces,_,_,_ = face_detector.detect_align(img)

    if faces.numel() == 0:
        return "No faces were found on this image, try with another one"


    genders, ages, emotions, probabilities = get_age_gender_emotion(faces,
     age_gender_detector,
      emotion_detector)

    create_images_age_gender_emotion(faces, ages, genders, emotions, probabilities)
    create_face_segregation_images(faces,"A.G.E.")

    return render_template("a_g_e_estimator.html",
        estimations='estimations.svg',
        face_segregation='face_segregation.svg')
    



if __name__ == "__main__":
    app.run(debug = True)
    

