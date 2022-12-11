
from ariadne import MutationType
from api.types import * 
from api.models.pytorch import *
from api.models import *
from PIL import Image
import io

mutation = MutationType()
@mutation.field("predictAnimal")
def predict_animal_resolver(obj, info, input):
   try:
        ext = "."+str(input['image'].filename).split('.')[-1]
        if not ext in allowed_extensions:
            return {
                "ok" : False,
                "error": {
                    "field" : 'image',
                    "message" : f'Only images with extensions ({", ".join(allowed_extensions)}) are allowed.'
                }
            }
        image = input['image'].read()
        image = Image.open(io.BytesIO(image))
        tensor = preprocess_img(image)
        res = predict(air_model, tensor, device)
        return {
           'ok': True,
           'prediction': res.to_json() 
        }
   except Exception as e:
       print(e)
       return {
           "ok": False,
           "error":{
               "field": 'server',
               'message':  "Something went wrong on the server."
           }
       }