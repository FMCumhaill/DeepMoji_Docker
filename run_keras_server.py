# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
from attention_layer import AttentionWeightedAverage
from keras.models import load_model
import json
from sentence_tokenizer import SentenceTokenizer
import pandas as pd

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model_1():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	#model = ResNet50(weights="imagenet")
	"""
	.h5 created externally and sent here. The attentionlayer from .py 
	"""
	model = load_model('emoji.h5',custom_objects={'AttentionWeightedAverage':AttentionWeightedAverage},compile=True)
	
	global st 
	with open("vocabulary.json", 'r') as f:
		vocabulary = json.load(f)
	st = SentenceTokenizer(vocabulary, 30)

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


@app.route("/predict/<text_input>")
def predict(text_input):
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	#if flask.request.method == "POST":
		
		#if flask.request.files.get("image"):
	#text_input = flask.request.files["image"].read()
			#print(image)
			#TEST_SENTENCES = [u'I love mom\'s cooking']
			#print('Tokenizing using dictionary from {}'.format("vocabulary.json"))
	
			#text_input = unicode(text_input,"utf-8")
	text_input = str(text_input)
	text_input = unicode(text_input)
			
	tokenized, _, _ = st.tokenize_sentences([text_input])
	pred = model.predict(tokenized)
			#for i, t in enumerate(text_input):
			#	t_tokens = tokenized[i]
			#	t_score = [t]
			#	t_prob = pred[i]
			#	ind_top = top_elements(t_prob, 3)
	values_list = top_elements(pred[0], 3)
	values = values_list[0]
	values = str(values) + "_" + str(values_list[1]) + "_" + str(values_list[2])
	
	##Find other value 
	
	#data['values'] = str(values + " " + Target) 
	print(text_input,values + " " + values)
	return '{}\''.format(values + " " + values)


	#data['predictions'] = pred
	#preds = model.predict(TEST_SENTENCES)
	#print(preds)
	#results = imagenet_utils.decode_predictions(preds)
	#data["predictions"] = ["hello"]

	# loop over the results and add them to the list of
	# returned predictions
	#for (imagenetID, label, prob) in results[0]:
	#	r = {"label": label, "probability": float(prob)}
	#	data["predictions"].append(r)

	# indicate that the request was a success
	#data["success"] = True
			
		
	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model_1()
	#predict()
	app.run(host='0.0.0.0', port="80")