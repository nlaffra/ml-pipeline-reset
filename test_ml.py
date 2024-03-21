import pytest, logging, os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.model import inference
from ml.data import process_data
@pytest.fixture(scope="module")
def data():
    # Fixture - returns dataset 
    datapath = "./data/census.csv"
    return pd.read_csv(datapath)

@pytest.fixture(scope="module")
def features():
    """
    Fixture - returns features
    """
    cat_features = [    "workclass",
                        "education",
                        "marital-status",
                        "occupation",
                        "relationship",
                        "race",
                        "sex",
                        "native-country"]
    return cat_features

@pytest.fixture(scope="module")
def train_dataset(data, features):
    """
    Fixture - split data set and return train for testing
    """
    train, test = train_test_split( data, 
                                test_size=0.20, 
                                random_state=10, 
                                stratify=data['salary']
                                )
    X_train, y_train, encoder, lb = process_data(
                                            train,
                                            categorical_features=features,
                                            label="salary",
                                            training=True
                                        )
    return X_train, y_train
    
def test_is_model():
    """
    Check saved model exists
    """
    savepath = "./model/model.pkl"
    if os.path.isfile(savepath):
        try:
            _ = pickle.load(open(savepath, 'rb'))
        except Exception as err:
            logging.error(
            "Testing saved model: Invalid saved model")
            raise err
    else:
        pass

def test_features(data, features):
    """
    Check that categorical features exist in dataset
    """
    try:
        assert sorted(set(data.columns).intersection(features)) == sorted(features)
    except AssertionError as err:
        logging.error(
        "Testing dataset: Features are missing in the data columns")
        raise err

def test_inference(train_dataset):
    """
    Check that inference function works
    """
    X_train, y_train = train_dataset

    savepath = "./model/trained_model.pkl"
    if os.path.isfile(savepath):
        model = pickle.load(open(savepath, 'rb'))

        try:
            preds = inference(model, X_train)
        except Exception as err:
            logging.error(
            "Inference cannot be performed on saved model and train data")
            raise err
    else:
        pass
