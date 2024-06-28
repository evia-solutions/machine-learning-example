"""
The source code and the article could be found
here : https://towardsdatascience.com/a-gentle-introduction-to-self-training-and-semi-supervised-learning-ceee73178b38
Note:
    1. Some lines are intentionally commented because generate a verbose output.
    2. The same example could be run at Jupyter notebook using the self-learning.ioynb file.
"""

import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile
from sklearn.linear_model import LogisticRegression
import os
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay

__path = '../../resources/data/Surgical-deepnet.zip'
__member = 'Surgical-deepnet.csv'


def load_dataset():
    # Load data
    with ZipFile(__path, 'r') as zip_file:
        surgical_deenet_file = zip_file.extract(__member)
    df = pd.read_csv(surgical_deenet_file)
    # Get more informantions about the dataset.
    # df.info()
    os.remove(surgical_deenet_file)
    return df


def preparing_dataset(df):
    # Shuffle the data
    df = df.sample(frac=1, random_state=15).reset_index(drop=True)

    # Generate indices for splits
    test_ind = round(len(df) * 0.25)
    train_ind = test_ind + round(len(df) * 0.01)
    unlabeled_ind = train_ind + round(len(df) * 0.74)

    # Partition the data
    test = df.iloc[:test_ind]
    train = df.iloc[test_ind:train_ind]
    unlabeled = df.iloc[train_ind:unlabeled_ind]

    # Assign data to train, test, and unlabeled sets
    X_train = train.drop('complication', axis=1)
    y_train = train.complication
    X_unlabeled = unlabeled.drop('complication', axis=1)
    X_test = test.drop('complication', axis=1)
    y_test = test.complication

    # Check dimensions of data after splitting
    print("Check dimensions of data after splitting")
    print("----------------------------------------")
    print(f"X_train dimensions: {X_train.shape}")
    print(f"y_train dimensions: {y_train.shape}\n")
    print(f"X_test dimensions: {X_test.shape}")
    print(f"y_test dimensions: {y_test.shape}\n")
    print(f"X_unlabeled dimensions: {X_unlabeled.shape}")

    # Visualize class distribution
    y_train.value_counts().plot(kind='bar')
    plt.xticks([0, 1], ['No Complication', 'Complication'])
    plt.ylabel('Count')
    return X_train, y_train, X_test, y_test, X_unlabeled


def train_labeled_data(X_train, y_train, X_test, y_test):
    # Logistic Regression Classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_hat_test = clf.predict(X_test)
    y_hat_train = clf.predict(X_train)
    train_f1 = f1_score(y_train, y_hat_train)
    test_f1 = f1_score(y_test, y_hat_test)

    print(f"Train f1 Score: {train_f1}")
    print(f"Test f1 Score: {test_f1}")

    disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, cmap=plt.cm.Blues, normalize='true',
                                                 display_labels=['No Comp.', 'Complication']);
    print("Confusion matrix")
    print(disp.confusion_matrix)
    # Generate probabilities for each prediction
    clf.predict_proba(X_test)
    return clf


def generate_pseudo_label(clf, X_train, y_train, X_test, y_test, X_unlabeled):
    # Initiate iteration counter
    iterations = 0

    # Containers to hold f1_scores and # of pseudo-labels
    train_f1s = []
    test_f1s = []
    pseudo_labels = []

    # Assign value to initiate while loop
    high_prob = [1]

    # Loop will run until there are no more high-probability pseudo-labels
    while len(high_prob) > 0:
        # Fit classifier and make train/test predictions
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_hat_train = clf.predict(X_train)
        y_hat_test = clf.predict(X_test)

        # Calculate and print iteration # and f1 scores, and store f1 scores
        train_f1 = f1_score(y_train, y_hat_train)
        test_f1 = f1_score(y_test, y_hat_test)
        # print(f"Iteration {iterations}")
        # print(f"Train f1: {train_f1}")
        # print(f"Test f1: {test_f1}")
        train_f1s.append(train_f1)
        test_f1s.append(test_f1)

        # Generate predictions and probabilities for unlabeled data
        # print(f"Now predicting labels for unlabeled data...")

        pred_probs = clf.predict_proba(X_unlabeled)
        preds = clf.predict(X_unlabeled)
        prob_0 = pred_probs[:, 0]
        prob_1 = pred_probs[:, 1]

        # Store predictions and probabilities in dataframe
        df_pred_prob = pd.DataFrame([])
        df_pred_prob['preds'] = preds
        df_pred_prob['prob_0'] = prob_0
        df_pred_prob['prob_1'] = prob_1
        df_pred_prob.index = X_unlabeled.index

        # Separate predictions with > 99% probability
        high_prob = pd.concat([df_pred_prob.loc[df_pred_prob['prob_0'] > 0.99],
                               df_pred_prob.loc[df_pred_prob['prob_1'] > 0.99]],
                              axis=0)
        # print(f"{len(high_prob)} high-probability predictions added to training data.")
        pseudo_labels.append(len(high_prob))

        # Add pseudo-labeled data to training data
        X_train = pd.concat([X_train, X_unlabeled.loc[high_prob.index]], axis=0)
        y_train = pd.concat([y_train, high_prob.preds])

        # Drop pseudo-labeled instances from unlabeled data
        X_unlabeled = X_unlabeled.drop(index=high_prob.index)
        # print(f"{len(X_unlabeled)} unlabeled instances remaining.\n")

        # Update iteration counter
        iterations += 1
    # Plot f1 scores and number of pseudo-labels added for all iterations
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
    ax1.plot(range(iterations), test_f1s)
    ax1.set_ylabel('f1 Score')
    ax2.bar(x=range(iterations), height=pseudo_labels)
    ax2.set_ylabel('Pseudo-Labels Created')
    ax2.set_xlabel('# Iterations');
    return clf, X_test, y_test


def plot_confusion_matrix(clf, X_test, y_test):
    # View confusion matrix after self-training
    disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, cmap=plt.cm.Blues, normalize='true',
                                                 display_labels=['No Comp.', 'Complication']);
    print("Confusion matrix")
    print(disp.confusion_matrix)
    plt.show()


def main():
    df = load_dataset()
    X_train, y_train, X_test, y_test, X_unlabeled = preparing_dataset(df)
    clf = train_labeled_data(X_train, y_train, X_test, y_test)
    clf, X_test, y_test = generate_pseudo_label(clf, X_train, y_train, X_test, y_test, X_unlabeled)
    plot_confusion_matrix(clf, X_test, y_test)


if __name__ == "__main__":
    main()
