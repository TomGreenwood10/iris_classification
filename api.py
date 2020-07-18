from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier


class Data(BaseModel):
    values: List[List[float]]


# Only one model so load at start ready for requests later
with open('model.pickle', 'rb') as f:
    model: KNeighborsClassifier = pickle.load(f)

app = FastAPI()


@app.post('/predict')
def predict(data: Data):
    X = np.array(data.values)
    prediction = model.predict(X)
    response = {
        'prediction': prediction.tolist()
    }
    return response
