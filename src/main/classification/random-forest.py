from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris


def execute_random_florest():
    # Load data
    data = load_iris()
    X = data.data
    y = data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a classifier
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    # Predict
    predictions = classifier.predict(X_test)
    # Perform cross-validation
    scores = cross_val_score(classifier, X, y, cv=5)
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
    execute_random_florest()
