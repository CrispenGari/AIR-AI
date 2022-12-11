import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from api.app import app
from flask import make_response, jsonify, request, json
from api.blueprints import blueprint
from ariadne import  load_schema_from_path, make_executable_schema, graphql_sync, upload_scalar, combine_multipart_data
from ariadne.constants import PLAYGROUND_HTML
from api.resolvers.queries import query
from api.resolvers.mutations import mutation

type_defs = load_schema_from_path("schema/schema.graphql")
schema = make_executable_schema(
    type_defs, [upload_scalar, query, mutation, ]
)
app.register_blueprint(blueprint, url_prefix="/api")


class AppConfig:
    PORT = 3001
    DEBUG = False
    
@app.route('/', methods=["GET"])
def meta():
    meta = {
        "programmer": "@crispengari",
        "main": "Animal Image Recognition (AIR)",
        "description": "given an image of an animal, the API should be able to predict the name of an animal among the 10 animals.",
        "language": "python",
        "libraries": ["pytorch", "torchvision"],
    }
    return make_response(jsonify(meta)), 200

@app.route("/graphql", methods=["GET"], )
def graphql_playground():
    return PLAYGROUND_HTML, 200

@app.route("/graphql", methods=["POST"])
def graphql_server():
    if  request.content_type.startswith("multipart/form-data" ):
         data = combine_multipart_data(
            json.loads(request.form.get("operations")),
            json.loads(request.form.get("map")),
            dict(request.files)
        )
    else:
        data =  request.get_json()
    success, result = graphql_sync(
        schema,
        data,
        context_value=request,
        debug= AppConfig.DEBUG
    )
    return jsonify(result), 200 if success else 400

if __name__ == "__main__":
    app.run(debug=AppConfig().DEBUG, port=AppConfig().PORT, )