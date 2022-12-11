from ariadne import QueryType
from api.types import * 


query = QueryType()
@query.field("meta")
def meta_resolver(obj, info):
   return Meta(
        programmer = "@crispengari",
        main = "Animal Image Recognition (AIR)",
        description = "given an image of an animal, the API should be able to predict the name of an animal among the 10 animals.",
        language = "python",
        libraries = ["pytorch", "torchvision"],
   ).to_json()
   