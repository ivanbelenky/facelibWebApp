from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from facelib import FaceDetector

import sys

def create_face_segregation_images(faces, route = "detection"):

    titles = [f"Face{i+1}" for i in range(faces.shape[0])]
    
    fig = make_subplots( rows=1, cols=faces.shape[0], subplot_titles=titles)
    fig.update_layout(title_text="Face Detection")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    print(faces.shape)
    for i,face in enumerate(faces):    
        fig.add_trace(px.imshow(face.detach().numpy()).data[0], row=1, col=i+1)

    fig.update_layout(autosize=False, width=5*faces.shape[0]*100,height=400)
    
    #fig.write_image("testfacesegmentation.svg", engine='kaleido')
    if route == "A.G.E.":
        fig.write_image(sys.path[0]+"/processed/age_gender_emotion/face_segregation.svg")
        return 

    fig.write_image(sys.path[0]+"/processed/detection/face_segregation.svg")
    return


def create_face_landmarks_images(img,faces, boxes, scores, landmarkss):
    titles = [f"Face{i+1}" for i in range(faces.shape[0])]

    fig = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="LandMarks: Facial Expression markers")))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.add_trace(px.imshow(img).data[0])
    
    idx = 0
    for face,box,landmarks in zip(faces,boxes,landmarkss):
        #rectangle
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))   
        
        fig.add_trace(go.Scatter(
            x=[(c1[0]+c2[0])/2],
            y=[(c1[1]+c2[1])/2],
            text=titles[idx],
            mode="text",
            name=titles[idx]
        ))
        fig.add_shape(type="rect",
            x0=c1[0], y0=c1[1], x1=c2[0], y1=c2[1],
            line=dict(
                width=2,
            ),
            name=titles[idx]
        )

        #landmarks
        fig.add_trace(go.Scatter(x=landmarks[:,0], y=landmarks[:,1],mode="markers",
        text=["left eye","right eye","nose","left mouth end","right mouth end"],name=f"Landmarks for {titles[idx]}"))
        
        idx += 1

    fig.write_image(sys.path[0]+"/processed/detection/face_markers.svg")

    return 