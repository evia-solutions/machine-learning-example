"""
This code is based on the article: https://towardsdatascience.com/encoding-categorical-data-explained-a-visual-guide-with-code-example-for-beginners-b169ac4193ae
"""
import pandas as pd
import numpy as np


def create_data():
    """"Create a DataFrame from the dictionary."""
    data = {
        'Date': ['03-25', '03-26', '03-27', '03-28', '03-29', '03-30', '03-31', '04-01', '04-02', '04-03', '04-04', '04-05'],
        'Weekday': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
        'Month': ['Mar', 'Mar', 'Mar', 'Mar', 'Mar', 'Mar', 'Mar', 'Apr', 'Apr', 'Apr', 'Apr', 'Apr'],
        'Temperature': ['High', 'Low', 'High', 'Extreme', 'Low', 'High', 'High', 'Low', 'High', 'Extreme', 'High', 'Low'],
        'Humidity': ['Dry', 'Humid', 'Dry', 'Dry', 'Humid', 'Humid', 'Dry', 'Humid', 'Dry', 'Dry', 'Humid', 'Dry'],
        'Wind': ['No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'],
        'Outlook': ['sunny', 'rainy', 'overcast', 'sunny', 'rainy', 'overcast', 'sunny', 'rainy', 'sunny', 'overcast', 'sunny', 'rainy'],
        'Crowdedness': [85, 30, 65, 45, 25, 90, 95, 35, 70, 50, 80, 45]
    }
    return pd.DataFrame(data)


def create_label_encoding(data_frame, source_column_name, new_column_name):
    """This type of encoding is used when is a clear order in the categories. It assign a unique integer value to each
     category in the categorical variable.
     e.g: product rating: 1 star, 2 stars 3 stars

     :data_frame pd.DataFrame: Data used in the example
     :target_column_name str:  Name of the column that will be used to generate the encoding.
     :new_column_name str:  Name of the encoded column in the data frame.
     """
    data_frame[new_column_name] = pd.factorize(data_frame[source_column_name])[0]


def create_one_hot_encoding(data_frame, source_column_name, prefix_for_new_column):
    """One-hot encoding creates a binary column for each category in the categorical value.
    Common use: It's used in nominal variables where there's no order between the values. It's useful for
    categories that have a small number of categories.
    e.g: sunny 100, overcast 010 and rainy 001

    :data_frame pd.DataFrame: Data used in the example
    :source_column_name str: Name of the column in the data frame that will be used as data source for the transformation
    :prefix_for_new_column str: Name of the prefix of the new column.
    """
    return pd.get_dummies(data_frame, columns=[source_column_name], prefix=prefix_for_new_column)


def create_binary_encoding(data_frame, source_column_name, binary_value, new_column_name):
    """Binary encoding transform a categorical value in a binary number (0,1). It used when the categorical value has 2 values, mostly
     yes/no.
     Common use in situation where the categorical values present 2 values like true/false, yes/no.

    :data_frame pd.DataFrame: Data used in the example
    :original_column_name  str: Name of the the column used in the encoding.
    :binary_value str: Value used to compare in each row of the data frame.
    :new_column_name str: Name of the new column that holds the encoding results.
     """
    data_frame[new_column_name] = (data_frame[source_column_name] == binary_value).astype(int)


def create_target_encoding(data_frame, source_group_column_name, target_column_name, new_column_name):
    """Target encoding replace categorical values by the mean of the target value.
    Common use:  It's used when the categorical values are high-cardinality feature in dataset with reasonable rows.
    By high-cardinality means columns with values that are very uncommon or unique.

    :data_frame pd.DataFrame: Data used in the example
    :source_group_column_name str: Name of the column that will be used to create the groups to calculate the mean
    :target_column_name str: Name of the target column which the values will be used in the mean value calculation.
    :new_column_name str: Name of new column that will contain the encoded value.
    """
    data_frame[new_column_name] = data_frame.groupby(source_group_column_name)[target_column_name].transform('mean')


def create_ordinal_encoding(data_frame, source_column_name, new_column_name):
    """Ordinal encoded is used when you have a categorical values that has inherent order and this order should be preserved.
    Some examples are:
    1. Heat level : warm, hot, very hot
    2. Feeling: good, bad, so bad, neutral
    3. Sleeve Length: long, short, sleevesless

    :data_frame pd.DataFrame: Data used in the example
    :source_column_name str: Name of the source column which the values will be encoded.
    :new_column_name str: Name of the new column that will hold the encoded value.
    """
    temp_order = {'Low': 1, 'High': 2, 'Extreme': 3} # mapping
    data_frame[new_column_name] = data_frame[source_column_name].map(temp_order)


def create_cyclic_encoding(data_frame, source_column_name, source_column_as_num_name, new_column_sin_name, new_column_cos_name):
    """Cyclic encoded is used for categorical values that has a clear cycle order, and this cycle should be preserved. To preserve the
    cycle nature is applied cosine and sine transformations. Examples of this kind of categorical values are:
    1. days of week
    2. months of year
    3. hours of day

    It is particularly useful when the 'distance' between categories matters and wraps around."""
    month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                       'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    data_frame[source_column_as_num_name] = data_frame[source_column_name].map(month_order)
    data_frame[new_column_sin_name] = np.sin(2 * np.pi * data_frame[source_column_as_num_name] / 12)
    data_frame[new_column_cos_name] = np.cos(2 * np.pi * data_frame[source_column_as_num_name] / 12)


def main():
    df = create_data()

    # 1. Label Encoding for Weekday
    create_label_encoding(df, 'Weekday', 'Weekday_label')

    # 2. One-Hot Encoding for Outlook
    df = create_one_hot_encoding(df, 'Outlook', 'Outlook')

    # 3. Binary Encoding for Wind
    create_binary_encoding(df, 'Wind', 'Yes', 'Wind_binary')

    # 4. Target Encoding for Humidity
    create_target_encoding(df, 'Humidity', 'Crowdedness', 'Humidity_target')

    # 5. Ordinal Encoding for Temperature
    create_ordinal_encoding(df, 'Temperature', 'Temperature_ordinal')

    # 6. Cyclic Encoding for Month
    create_cyclic_encoding(df, 'Month', 'Month_num', 'Month_sin', 'Month_cos')

    # Select and rearrange numerical columns
    numerical_columns = [
        'Date','Weekday_label',
        'Month_sin', 'Month_cos',
        'Temperature_ordinal',
        'Humidity_target',
        'Wind_binary',
        'Outlook_sunny', 'Outlook_overcast', 'Outlook_rainy',
        'Crowdedness'
    ]

    # Display the rearranged numerical columns
    print(df[numerical_columns].round(3))


if __name__ == "__main__":
    main()