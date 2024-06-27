from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import os


def execute_decision_tree_iris_ds():
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    predictions = clf.predict(X_test)

    # Evaluate
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Accuracy  :", accuracy_score(y_test, predictions))
    print("Precision :", precision_score(y_test, predictions, average='macro'))
    print("Recall    :", recall_score(y_test, predictions, average='macro'))
    print("F1 Score  :", f1_score(y_test, predictions, average='macro'))


def execute_decision_tree_wine_ds():
    wine = load_wine()
    X = wine.data
    y = wine.target
    output_path = "output"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_file = os.path.join(output_path,  "plot.png" )


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Splitting the dataset into training and testing sets

    clf = DecisionTreeClassifier() # Initialize the decision tree classifier

    clf.fit(X_train, y_train) # Fitting the classifier on the training data

    # Plot the decision tree
    plt.figure(figsize=(15, 10))
    plot_tree(clf, filled=True, feature_names=wine.feature_names, class_names=wine.target_names)
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()



def main():
    print("Decision tree with iris data set")
    execute_decision_tree_iris_ds()
    print("Decision tree with wine data set")
    execute_decision_tree_wine_ds()


if __name__ == "__main__":
    main()
