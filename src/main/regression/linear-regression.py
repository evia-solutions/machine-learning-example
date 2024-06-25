import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing, load_diabetes


def execute_linear_regression():
    # Load data
    data = fetch_california_housing()
    X = data.data
    y = data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("MSE:", mean_squared_error(y_test, predictions))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
    print("R-squared:", r2_score(y_test, predictions))


def execute_linear_regression_diabetes():
    # Code source: Jaques Grobler
    # License: BSD 3 clause
    # Load the diabetes dataset
    diabetes_X, diabetes_y = load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Create linear regression object
    regr = LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    #print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    #print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))
    # Evaluate
    print("MAE:", mean_absolute_error(diabetes_y_test, diabetes_y_pred))
    print("MSE:", mean_squared_error(diabetes_y_test, diabetes_y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(diabetes_y_test, diabetes_y_pred)))
    print("R-squared:", r2_score(diabetes_y_test, diabetes_y_pred))


    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
    plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()


if __name__ == "__main__":
    execute_linear_regression()
    execute_linear_regression_diabetes()