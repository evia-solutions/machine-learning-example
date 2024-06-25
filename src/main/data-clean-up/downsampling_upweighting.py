"""
Downsampling: This involves reducing the size of the majority class in an imbalanced dataset to match the size of the minority class.
Upweighting: This involves increasing the importance (weight) of the minority class instances when training a model.

1. We create an imbalanced dataset using make_classification from sklearn.datasets.

2. For downsampling, we reduce the majority class size to match the minority class size using resample from sklearn.utils.

3. For upweighting, we assign higher weights to the minority class samples by specifying class_weight when training the
RandomForestClassifier.

4. Finally, we train models on both the downsampled dataset and the original dataset with upweighting and evaluate their
performance using classification_report from sklearn.metrics.
"""
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def separating_classes():
    # Create a synthetic imbalanced dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2,
                               n_redundant=10, n_clusters_per_class=1,
                               weights=[0.9, 0.1], flip_y=0, random_state=1)

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(X)
    df['target'] = y

    # Separate majority and minority classes
    df_majority = df[df.target == 0]
    df_minority = df[df.target == 1]
    return df_majority, df_minority, X, y


def downsampling(df_majority, df_minority):
    # Downsample majority class
    df_majority_downsampled = resample(df_majority,
                                       replace=False,    # sample without replacement
                                       n_samples=len(df_minority),     # to match minority class
                                       random_state=123) # reproducible results

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    # Split data into training and testing sets
    X_downsampled = df_downsampled.drop('target', axis=1)
    y_downsampled = df_downsampled['target']
    X_train_ds, X_test_ds, y_train_ds, y_test_ds = train_test_split(X_downsampled, y_downsampled, test_size=0.3, random_state=42)

    # Train model on downsampled data
    model_ds = RandomForestClassifier(random_state=42)
    model_ds.fit(X_train_ds, y_train_ds)
    y_pred_ds = model_ds.predict(X_test_ds)

    print("Downsampling results:\n")
    print(classification_report(y_test_ds, y_pred_ds))

def upweighting(X,y):
    # Upweighting example
    # Split original data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Set class weights for upweighting
    class_weights = {0: 1, 1: 10}

    # Train model with class weights
    model_uw = RandomForestClassifier(random_state=42, class_weight=class_weights)
    model_uw.fit(X_train, y_train)
    y_pred_uw = model_uw.predict(X_test)

    print("Upweighting results:\n")
    print(classification_report(y_test, y_pred_uw))


def main():
   result_classes = separating_classes()
   downsampling(result_classes[0],result_classes[1])
   upweighting(result_classes[2],result_classes[3])


if __name__ == "__main__":
    main()