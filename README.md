# Cook-E: Healthy cooking made easy

Cook-E is a mobile application that aims to simplify cooking by providing users with recipes based on the ingredients they have on hand and their health condition persona. With the current trend towards healthy eating, many people want to cook their meals at home using fresh ingredients.

## Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/id/)
- [Tensorflow](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/stable/)


### Documentation

[Documentation](https://cooke-ml-image-54vjhz35da-et.a.run.app/docs)

## Setup and Installation

1. Clone repository

```
git clone "https://github.com/Cook-E-Capstone/Cook-E-ML-Image.git"
```

2. Create virtual env Python

```
python -m venv env
```

3. Activate virtual env python

```
./env/Scripts/activate
```

4. Install Python depedencies

```
pip install -r requirements.txt
```

## Development Guide

### Non docker

1. Create environment variables in file .env

```
API_KEY= 
```

2. Run app in development environment

```
uvicorn app.main:app --reload
```
