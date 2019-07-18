

import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:80/predict"


# load the input image and construct the payload for the request
payload = {"image": "You love hurting me, huh?"}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was successful
if r["success"]:
    # loop over the predictions and display them
    #for (i, result) in enumerate(r["predictions"]):
    #    print("{}. {}: {:.4f}".format(i + 1, result["label"],
    #        result["probability"]))
   	print(r)

# otherwise, the request failed
else:
    print("Request failed")