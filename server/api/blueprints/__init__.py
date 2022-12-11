from flask import Blueprint, make_response, jsonify, request
import io
from PIL import Image
from api.models import *
from api.models.pytorch import *

blueprint = Blueprint("blueprint", __name__)

@blueprint.route('/v1/classify', methods=["POST"]) 
def classify_animal():
    data = {"success": False}
    if request.method == "POST":
        if request.files.get("image"):
            img = request.files.get("image")
            ext = "."+str(img.filename).split('.')[-1]
            if not ext in allowed_extensions:
                data["success"] = False
                data['error'] = f'Only images with extensions ({", ".join(allowed_extensions)}) are allowed.'
                # read the image in PIL format
            else:
                try:
                    image = request.files.get("image").read()
                    image = Image.open(io.BytesIO(image))
                    tensor = preprocess_img(image)
                    res = predict(air_model, tensor, device)
                    data["success"] = True
                    data["predictions"] = res.to_json()
                except Exception as e:
                    print(e)
                    data["error"] =  "Something went wrong on the server."
                    data["success"] = False
                    
    return make_response(jsonify(data)), 200
    
    