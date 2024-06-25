import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def generate_data_frame():
    # Generate a sample DataFrame
    np.random.seed(0)
    data = np.random.randn(1000, 1) * 20 + 50  # Normal distribution
    return pd.DataFrame(data, columns=['value'])


def calculate_Z_score(df):
    # Calculate Z-scores
    df['z_score'] = (df['value'] - df['value'].mean()) / df['value'].std()


def generate_outliers(df):
    # Define threshold
    threshold = 3

    # Identify outliers
    outliers = df[np.abs(df['z_score']) > threshold]
    print(outliers)

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1

    # Define thresholds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
    print(outliers)


def identify_by_using_isolation_forest(df):
    # Fit the model
    iso = IsolationForest(contamination=0.05)  # Set contamination to expected outliers ratio
    df['anomaly'] = iso.fit_predict(df[['value']])

    # Identify outliers
    outliers = df[df['anomaly'] == -1]
    print(outliers)


def identify_by_using_DBSCAN(df):
    # Fit the model
    dbscan = DBSCAN(eps=3, min_samples=2)
    df['anomaly'] = dbscan.fit_predict(df[['value']])

    # Identify outliers
    outliers = df[df['anomaly'] == -1]
    print(outliers)


def generating_plot(df):
    # Generate a box plot
    plt.boxplot(df['value'])
    plt.show()

    # Generate a scatter plot
    plt.scatter(range(len(df)), df['value'])
    plt.show()


def main():
    df = generate_data_frame()
    calculate_Z_score(df)
    generate_outliers(df)
    identify_by_using_isolation_forest(df)
    identify_by_using_DBSCAN(df)
    generating_plot(df)


if __name__ == "__main__":
    main()

