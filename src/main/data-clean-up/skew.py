import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew


def detect_skew():
    # Generate sample data
    np.random.seed(0)
    data = np.random.exponential(scale=2, size=1000)  # Positively skewed data

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['value'])

    # Calculate skewness
    skewness = skew(df['value'])
    print(f'Skewness: {skewness}')

    # Plot histogram
    plt.hist(df['value'], bins=30, edgecolor='k')
    plt.title('Histogram of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    # Plot box plot
    plt.boxplot(df['value'], vert=False)
    plt.title('Box Plot of Data')
    plt.xlabel('Value')
    plt.show()


if __name__ == "__main__":
    # Will generate a right skewed data that could be checked on the histogram.
    detect_skew()
