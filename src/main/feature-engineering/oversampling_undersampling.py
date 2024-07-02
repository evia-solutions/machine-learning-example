"""
The article could be found here:
https://semaphoreci.medium.com/how-to-handle-imbalanced-data-for-machine-learning-in-python-b6d56c9a0489
"""
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def applying_undersampling():
    """
    Compares visually via histogram how the undersampling process works.
    :return: None
    """
    training_data, test_data = create_data()
    # Print the histogram of the initial classes
    plt.figure(figsize=(10, 6))
    plt.hist(test_data, bins=range(4), align='left', rwidth=0.8, color='blue', alpha=0.7)
    plt.title("Histogram of Initial Classes")
    plt.xlabel("Class")
    plt.ylabel("Number of Instances")
    plt.xticks(range(3), ['Class 0', 'Class 1', 'Class 2'])
    plt.show()

    # Apply undersampling using RandomUnderSampler
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(training_data, test_data)
    # Print the histogram of the resampled classes
    plt.figure(figsize=(10, 6))
    plt.hist(y_resampled, bins=range(4), align='left', rwidth=0.8, color='orange', alpha=0.7)
    plt.title("Histogram of Resampled Classes (Undersampling)")
    plt.xlabel("Class")
    plt.ylabel("Number of Instances")
    plt.xticks(range(3), ['Class 0', 'Class 1', 'Class 2'])
    plt.show()


def applying_oversampling():
    """
    Compares visually via histogram how the oversampling process works.
    :return: None
    """
    training_data, test_data = create_data()

    # Print the histogram of the initial classes
    plt.figure(figsize=(10, 6))
    plt.hist(test_data, bins=range(4), align='left', rwidth=0.8, color='blue', alpha=0.7)
    plt.title("Histogram of Initial Classes")
    plt.xlabel("Class")
    plt.ylabel("Number of Instances")
    plt.xticks(range(3), ['Class 0', 'Class 1', 'Class 2'])
    plt.show()

    # Apply oversampling using RandomOverSampler
    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(training_data, test_data)
    # Print the histogram of the resampled classes
    plt.figure(figsize=(10, 6))
    plt.hist(y_resampled, bins=range(4), align='left', rwidth=0.8, color='orange', alpha=0.7)
    plt.title("Histogram of Resampled Classes (Oversampling)")
    plt.xlabel("Class")
    plt.ylabel("Number of Instances")
    plt.xticks(range(3), ['Class 0', 'Class 1', 'Class 2'])
    plt.show()


def create_data():
    # Create an imbalanced dataset with 3 classes
    return make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.1, 0.3, 0.6],  # Class imbalance
        random_state=42
    )


def comparing_accuracy():
    """
    Compares the accuracy of the three results and we can see that there is a overfitting problem when
    we compare the training results with the test results.

    :return: None
    """
    training_data, test_data = create_data()
    # Split the original dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(training_data, test_data, test_size=0.2, random_state=42)
    # Apply oversampling using RandomOverSampler

    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)

    # Apply undersampling using RandomUnderSampler
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_train_undersampled, y_train_undersampled = undersampler.fit_resample(X_train, y_train)

    # Fit KNN classifier on the original train set
    knn_original = KNeighborsClassifier(n_neighbors=5)
    knn_original.fit(X_train, y_train)

    # Fit KNN classifier on the oversampled train set
    knn_oversampled = KNeighborsClassifier(n_neighbors=5)
    knn_oversampled.fit(X_train_oversampled, y_train_oversampled)

    # Fit KNN classifier on the undersampled train set
    knn_undersampled = KNeighborsClassifier(n_neighbors=5)
    knn_undersampled.fit(X_train_undersampled, y_train_undersampled)

    # Make predictions on train sets
    y_train_pred_original = knn_original.predict(X_train)
    y_train_pred_oversampled = knn_oversampled.predict(X_train_oversampled)
    y_train_pred_undersampled = knn_undersampled.predict(X_train_undersampled)
    # Make predictions on test sets
    y_test_pred_original = knn_original.predict(X_test)
    y_test_pred_oversampled = knn_oversampled.predict(X_test)
    y_test_pred_undersampled = knn_undersampled.predict(X_test)

    # Calculate and print accuracy for train sets
    print("Accuracy on Original Train Set:", accuracy_score(y_train, y_train_pred_original))
    print("Accuracy on Oversampled Train Set:", accuracy_score(y_train_oversampled, y_train_pred_oversampled))
    print("Accuracy on Undersampled Train Set:", accuracy_score(y_train_undersampled, y_train_pred_undersampled))

    # Calculate and print accuracy for test sets
    print("\nAccuracy on Original Test Set:", accuracy_score(y_test, y_test_pred_original))
    print("Accuracy on Oversampled Test Set:", accuracy_score(y_test, y_test_pred_oversampled))
    print("Accuracy on Undersampled Test Set:", accuracy_score(y_test, y_test_pred_undersampled))

def main():
    applying_oversampling()
    applying_undersampling()
    comparing_accuracy()


if __name__ == "__main__":
    main()