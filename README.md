### Animal Image Recognition (AIR)

`AIR` is a GraphQL and REST built with `flask` and `ariadne` to perform image classification based on `10` animals.

```
- Cat
- Cow
- Dog
- Elephant
- Gorilla
- Hippo
- Monkey
- Panda
- Tiger
- Zebra
```

<p align="center">
<img src="logo.png" alt="logo" width="30%"/>
<p>

Animal Image Recognition is an AI api for performing image classification task based on `10` animals that were used to train the model.

<p align="center">
<img src="cover.jpg" alt="cover" width="100%"/>
<p>

### AIR Tool (mobile)

AIR tool is an AI tool for mobile application that is used to perform some basic animal image classification task in real-time. Given an image from a camera or an imported image the application must be able to classify the name or class of the animal on that picture in real-time.

<p align="center">
<img src="1.jpeg" alt="cover" width="200"/>
<img src="2.jpeg" alt="cover" width="200"/>
<img src="3.jpeg" alt="cover" width="200"/>
<img src="0.jpeg" alt="cover" width="200"/>
<p>

### API

AIR api is a simple rest api that is served at `http://localhost:3001/api/v1/classify` and is able to the name of an animal from an image.

### API response

If a proper `POST` request is sent to the server at `http://127.0.0.1:3001/api/v1/classify` we will be able to get `~80%` accurate predictions of the name of the animal on that image.

### cURL request

If a `cURL` request is send to the server at localhost which looks as follows:

```shell
cURL -X POST -F image=@cat.jpeg http://127.0.0.1:3001/api/v1/classify
```

The server will respond with the `API` response which looks as follows:

```json
{
  "predictions": {
    "predictions": [
      {
        "class_name": "Cat",
        "label": 0,
        "probability": 0.9900000095367432
      },
      {
        "class_name": "Cow",
        "label": 1,
        "probability": 0.0
      },
      {
        "class_name": "Dog",
        "label": 2,
        "probability": 0.009999999776482582
      },
      {
        "class_name": "Elephant",
        "label": 3,
        "probability": 0.0
      },
      {
        "class_name": "Gorilla",
        "label": 4,
        "probability": 0.0
      },
      {
        "class_name": "Hippo",
        "label": 5,
        "probability": 0.0
      },
      {
        "class_name": "Monkey",
        "label": 6,
        "probability": 0.0
      },
      {
        "class_name": "Panda",
        "label": 7,
        "probability": 0.0
      },
      {
        "class_name": "Tiger",
        "label": 8,
        "probability": 0.0
      },
      {
        "class_name": "Zebra",
        "label": 9,
        "probability": 0.0
      }
    ],
    "top_prediction": {
      "class_name": "Cat",
      "label": 0,
      "probability": 0.9900000095367432
    }
  },
  "success": true
}
```

### GraphQL endpoint

GraphQL endpoint is served at `http://localhost:3001/graphql` sending a graphql request at this endpoint with an image file for example as follows using `cURL`:

```shell
curl http://localhost:3001/graphql -F operations='{ "query": "mutation ClassifyAnimal($input: AnimalInput!) { predictAnimal(input: $input) { error { field message } ok prediction { topPrediction {  label probability className } predictions {  label probability className } } } }", "variables": { "input": {"image": null} } }'  -F map='{ "0": ["variables.input.image"] }'  -F 0=@cat.jpeg
```

Will yield the results that looks as follows:

```json
{
  "data": {
    "predictAnimal": {
      "error": null,
      "ok": true,
      "prediction": {
        "predictions": [
          {
            "className": "Cat",
            "label": 0,
            "probability": 0.9900000095367432
          },
          {
            "className": "Cow",
            "label": 1,
            "probability": 0.0
          },
          {
            "className": "Dog",
            "label": 2,
            "probability": 0.009999999776482582
          },
          {
            "className": "Elephant",
            "label": 3,
            "probability": 0.0
          },
          {
            "className": "Gorilla",
            "label": 4,
            "probability": 0.0
          },
          {
            "className": "Hippo",
            "label": 5,
            "probability": 0.0
          },
          {
            "className": "Monkey",
            "label": 6,
            "probability": 0.0
          },
          {
            "className": "Panda",
            "label": 7,
            "probability": 0.0
          },
          {
            "className": "Tiger",
            "label": 8,
            "probability": 0.0
          },
          {
            "className": "Zebra",
            "label": 9,
            "probability": 0.0
          }
        ],
        "topPrediction": {
          "className": "Cat",
          "label": 0,
          "probability": 0.9900000095367432
        }
      }
    }
  }
}
```

### Notebooks

The notebooks for training the model that is being used to perform image classification on animals is found [here](https://github.com/CrispenGari/cv-torch/blob/main/03_ANIMAL_IMAGE_RECOGNITION/01_ANIMAL_IMAGE_RECOGNITION.ipynb).

### License

In this simple AI tool i'm using `MIT` license which read as follows:

```shell
MIT License

Copyright (c) 2022 crispengari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
