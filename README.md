# iris_classification
API for classification of iris data from the classic 'iris' dataset.

The API is provided by fastAPI.

## How to use
### Start the server
- Install all the dependencies with `pip install -r requirements.txt`
- Start the server with `uvicorn api:app`
### Send post request to API
Requestes should be sent to `http://127.0.0.1:8000/predict` in the following json format:

```
{
    "values": <feature matrix>
}
```
With the feature matrix a list of lists containing the features: 

['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)']

If you have a feature matrix in a pandas DataFrame (with feature columns in the order above) you can produce the list with `pandas.DataFrame.to_numpy().tolist()`
## Whats in the repo
This repo is to demonstrate the serving of an ML model with FastAPI (which is great).  Any model with a predict() method could be saved inplace of `model.pickle`.  I decided to use the classic 'iris' dataset (in scikit-learn) and there is a jupyter notebook with EDA and model optimisation in `iris.ipynb`.

**Contents**
- `api.py` -- contains the web app and data processing
- `model.pickle` -- the binarised sklearn.neighbors.KNeighborsClassifier trained model
- `requirements.txt` -- the usual; all required packages (for the model type contained)
- `iris.ipynb` -- jupyer notebook with EDA and model tuning