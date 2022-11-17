from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from flask import Flask, redirect, render_template, request
from keras.applications.inception_v3 import preprocess_input
from keras.models import model_from_json
from werkzeug.utils import secure_filename

import secrets 
from flask import Flask, flash, render_template, request, redirect, url_for
from flask_uploads import IMAGES, UploadSet, configure_uploads

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
print(ROOT_DIR)

global graph
graph=tf.compat.v1.get_default_graph()
predictions = ["Corpse Flower", 
               "Great Indian Bustard", 
               "Lady's slipper orchid", 
               "Pangolin", 
               "Spoon Billed Sandpiper", 
               "Seneca White Deer"
              ]
             
found = [
        "https://en.wikipedia.org/wiki/Amorphophallus_titanum",
        "https://en.wikipedia.org/wiki/Great_Indian_bustard",
        "https://en.wikipedia.org/wiki/Cypripedioideae",
        "https://en.wikipedia.org/wiki/Pangolin",
        "https://en.wikipedia.org/wiki/Spoon-billed_sandpiper",
        "https://en.wikipedia.org/wiki/Seneca_white_deer",
        ]

app = Flask(__name__)
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = ROOT_DIR+"\\Final Deliverables\\Uploads\\"
app.config["SECRET_KEY"] = str(secrets.SystemRandom().getrandbits(128))
configure_uploads(app, photos)

@app.route('/', methods=['GET'])

def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])

def upload():
    
    if request.method == 'GET':
        return ("<h6 style=\"font-face:\"Courier New\";\">No GET request herd.....</h6 >")
    
    if request.method == 'POST':

        photos.save(request.files["upload-btn"])     
        f = request.files['upload-btn']
        img = tf.keras.utils.load_img(ROOT_DIR+"\\Final Deliverables\\Uploads\\"+secure_filename(f.filename), target_size=(224, 224))
        x = tf.keras.utils.img_to_array(img)
        x = preprocess_input(x)
        inp = np.array([x])
        with graph.as_default():
            json_file = open('Final Deliverables\\DigitalNaturalist.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("Final Deliverables\\DigitalNaturalist.h5")
            preds =  np.argmax(loaded_model.predict(inp),axis=1)
            print("Predicted the Species " + str(predictions[preds[0]]))
        text = found[preds[0]]
        return redirect(text)



if __name__ == '__main__':
    app.run(threaded = True,debug=True,port="8000")
    
    
# D:\\Projects\\IBM\\Assesment\\IBM-Project-7059-1658846575\\Project Development Phase\\Final Deliverables\\DigitalNaturalist.json
# D:\Projects\IBM\Assesment\IBM-Project-7059-1658846575\Final Deliverables\DigitalNaturalist.json