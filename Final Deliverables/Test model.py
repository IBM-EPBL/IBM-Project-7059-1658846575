from __future__ import division, print_function
import numpy as np
import tensorflow as tf
from keras.applications.inception_v3 import preprocess_input
from keras.models import model_from_json

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


print("==========================================================")
img_path = input("Enter the test image path:- ")
print("==========================================================")

img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
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
    print("==========================================================")
    print("Predicted the Species - " + str(predictions[preds[0]]))
    print("Information of the Species - " + str(found[preds[0]]))
    print("==========================================================")
