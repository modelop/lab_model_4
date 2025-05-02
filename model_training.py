from copyreg import pickle

import shap
import pickle

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def main():
    # California Housing Prices
    dataset = fetch_california_housing(as_frame=True)
    X = dataset['data']
    y = dataset['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_test.to_csv("./test_data.csv")

    # Prepares a default instance of the random forest regressor
    model = RandomForestRegressor()

    # Fits the model on the data
    model.fit(X_train, y_train)
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    # Create the Shap explainer for the trained model
    explainer = shap.Explainer(model.predict, X_test)
    with open("shap_explainer.pkl", "wb") as explainer_file:
        pickle.dump(explainer, explainer_file)


if __name__ == '__main__':
    main()