from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score


def execute_logistic_regression():
    # Load data
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Create a logistic regression model
    model = LogisticRegression(solver='liblinear')  # 'liblinear' is good for small datasets

    # Fit the model
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate the model

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=5)
    print("****************************************************************************************")
    print("Accuracy scores for each fold:", scores)
    print("****************************************************************************************")
    # Evaluate
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Accuracy  :", accuracy_score(y_test, predictions))
    print("Precision :", precision_score(y_test, predictions, average='macro'))
    print("Recall    :", recall_score(y_test, predictions, average='macro'))
    print("F1 Score  :", f1_score(y_test, predictions, average='macro'))


if __name__ == "__main__":
    execute_logistic_regression()
