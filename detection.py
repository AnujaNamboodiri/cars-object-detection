import pandas as pd
import numpy as np
from util import utilsize
from PIL import Image
import time
import cv2
from keras.models import model_from_json
import tensorflow as tf
import matplotlib.pyplot as plt

def detection(filename, go):

    # load json and create model
    json_file = open('models/model2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/model2.h5")
    print("Loaded model from disk")

    classdf = pd.read_csv("names.csv", names=['Name'])
    classdf["Class"] = np.arange(0,196)
    classes = dict(zip(classdf['Name'], classdf['Class']))

    path = "static/uploads/"+filename
    if go:
        path = "static/sample/"+filename

    img = tf.keras.preprocessing.image.load_img(path)
    w, h = img.size
    img_size = 224

    fontthick,rectsize,fontscale,ymargin = utilsize(h,w)
    font = cv2.FONT_HERSHEY_PLAIN   

    key_list = list(classes.keys())
    val_list = list(classes.values())

    ## Prepare input for model

    #1. Resize image
    img_resized = img.resize((img_size, img_size)) 

    #2. Conver to array and make it a batch of 1
    input_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    input_array = np.expand_dims(input_array, axis=0)

    #3. Normalize image data
    input_array = tf.keras.applications.resnet_v2.preprocess_input(input_array)

    #Prediction
    pred = loaded_model.predict(input_array)

    #Get classification and regression predictions
    scores, bbox_pred = pred[0][0], pred[1][0]

    #Get Label with highest probability
    
##Changes made by Sriram for making Dataframe of Scores
    #class_id = np.argmax(scores)
    #confidence = scores[class_id] 
    pred_class = np.argmax(scores)
    pred_class1 = (scores)
    print('Real Label :', act_class, '\nPredicted Label: ', pred_class)
    carname = (classdf["Name"].iloc[pred_class])
    print("The subject Car belongs to",carname)
    #print("The Probability of the image being the", carname, 'is',((pred_class1[pred_class])*100))
    ranks = sorted( [(x,i) for (i,x) in enumerate(pred_class1)], reverse=True )
    ranks = ranks[0:3]
    print("Top probability values are as follows:")
    carnames = list()
    classv = list()
    probv = list()
    for x,y in ranks:
      carname = (classdf["Name"].iloc[y])
      carnames.append(carname)
      classv.append(y)
      x = (round(x, 3))*100
      probv.append(x)
    df = pd.DataFrame(list(zip(classv, carnames,probv)),columns =['Class', 'Carname', 'Score Values'])
    print(df)
    # print(class_id)
    # print(confidence)
##Changes made by Sriram for making Dataframe of Scores
    # Get the actual names for the given label
    pos = val_list.index(class_id)
    class_id = key_list[pos]


    x1 = int(bbox_pred[0] * w)
    y1 = int(bbox_pred[1] * h)
    x2 = int(bbox_pred[2] * w)
    y2 = int(bbox_pred[3] * h)
    
    label = str(classes[class_id])
    confidence = str(round(confidence,2))

    print(label)
    print(class_id)
    print(confidence)

    #Draw bounding boxes
    img = cv2.imread(path)

    img = cv2.rectangle(img, (x1, y1, x2, y2), (0,255,0), 2)
    img = cv2.putText(img, class_id + " " + confidence, (x1, y1 + ymargin), font, fontscale, (255,0,0), fontthick)

    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img1 = Image.fromarray(img1)
    img1.save("static/uploads/"+filename) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)

# detection(classes, filename = 'car1.jpg', go=True)
