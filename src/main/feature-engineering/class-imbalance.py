"""
 This python example could be found in the article:
 https://semaphoreci.medium.com/how-to-handle-imbalanced-data-for-machine-learning-in-python-b6d56c9a0489
"""
import matplotlib.pyplot as plt
import numpy as np


def check_imbalance_by_histogram(data):
    """
    Generates a graphic visualization in a form of histogram from an imbalance dataset
    :param data: The imbalance dataset.
    :return: None
    """
    # Create labels for the classes
    labels = np.concatenate((np.zeros(900), np.ones(100)))

    # Plot the class distribution
    plt.figure(figsize=(8, 6))
    plt.hist(data[labels == 0], bins=20, color='blue', alpha=0.6, label='Majority Class (Class 0)')
    plt.hist(data[labels == 1], bins=20, color='red', alpha=0.6, label='Minority Class (Class 1)')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.title('Class Distribution in an Imbalanced Dataset')
    plt.legend()
    plt.show()


def check_imbalance_by_barchart(data):
    """
    Generates a graphic visualization in a form of bar chart from an imbalance dataset
    :param data: The imbalance dataset
    :return: None
    """
    # Create labels for the classes
    labels = np.concatenate((np.zeros(900), np.ones(100)))

    # Count the frequencies of each class
    class_counts = [len(labels[labels == 0]), len(labels[labels == 1])]

    # Plot the class frequencies using a bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(['Majority Class (Class 0)', 'Minority Class (Class 1)'], class_counts, color=['blue', 'red'])
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.title('Class Frequencies in an Imbalanced Dataset')
    plt.show()


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate data for a majority class (Class 0)
    majority_class = np.random.normal(0, 1, 900)
    # Generate data for a minority class (Class 1)
    minority_class = np.random.normal(3, 1, 100)
    # Combine the majority and minority class data
    data = np.concatenate((majority_class, minority_class))

    check_imbalance_by_histogram(data)
    check_imbalance_by_barchart(data)


if __name__ == "__main__":
    main()