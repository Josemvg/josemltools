# Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Statistical libraries
import scipy.stats as st

# Other libraries
import pandas as pd 
import numpy as np

def study_column_continuous(df: pd.DataFrame, col: str, print_stats = True, plot = True, skew = True, gaussian_test = True) -> pd.DataFrame:
    """
    Performs a complete study of a column in a dataframe, for a continuous quantitative variable.
    - It prints the mean, median and standard deviation of the column.
    - It calculates the quantiles and gets the interquartile range and the lower and upper bounds for outliers.
    - It plots a boxplot with a scatterplot on top of it, to show the outliers.
    - It plots a histogram to show the distribution of the column, with a line showing the mean and the median.
    - It plots a QQ-plot to show the distribution of the column and does a normality test (Shapiro-Wilk test).
    - Finally, it builds a dataframe with the outliers and returns it.

    Parameters
    -------
    df: pandas dataframe
        The dataframe containing the column to be studied.
    col: string
        The name of the column to be studied.
    print_stats: boolean, optional
        If True, it prints the mean, median and standard deviation of the column.
    plot: boolean, optional
        If True, it plots the boxplot and a histogram.
    gaussian_test: boolean, optional
        If True, it plots a QQ-plot to show the distribution of the column and does a normality test (Shapiro-Wilk test).

    Returns
    --------
    outliers: pandas dataframe
        A dataframe containing the outliers of the column.
    """
    # Print the mean, median and standard deviation of the column
    if print_stats:
        print("The column ", col, " mean is ", df[col].mean())
        print("The column ", col, " median is ", df[col].median())
        print("The column ", col, " standard deviation is ", df[col].std())

    # Calculate the quantiles and get the interquartile range and the lower and upper bounds for outliers
    quants = []
    for i in range(1, 4):
        quants.append(df[col].quantile(i/4))
        print("Quantile ", i, " for ", col, " is ", quants[i-1])

    iqr = quants[2] - quants[0]
    print("IQR for the ", col, " variable: ", iqr)

    lowerBound = quants[0] - (1.5 * iqr)
    upperBound = quants[2] + (1.5 * iqr)
    print("Lower and upper outlier limits for the ", col, " variable:", lowerBound, ", ", upperBound)

    # Create a boxplot and a histogram to show the distribution of the column
    if plot:
        # Create a figure composed of two matplotlib.Axes objects (ax_box and ax_hist) with shared x-axis 
        fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.2, .8)})

        # Add a boxplot with a scatterplot on top of it on the ax_box object
        sns.boxplot(df[col], ax=ax_box, orient = "h", color = "tab:blue")
        sns.stripplot(df[col], ax=ax_box, alpha = 0.25, orient = "h", color='tab:red')

        # Add a histogram on the ax_hist object, with a line for the mean and median
        sns.histplot(df[col], ax=ax_hist,  bins=20, color = "tab:blue")
        ax_hist.axvline(df[col].mean(), 0,1,color='tab:red',label='Mean')
        ax_hist.axvline(df[col].median(), 0,1,color='tab:green',label='Median')

        #Add the vertical lines to the legend
        ax_hist.legend()
        # Set the column name as the x-axis label
        ax_box.set(xlabel=col)
        # Show the plot
        plt.show()

    # Study the skewness of the variable 
    if skew:
        skew = df[col].skew()
        print("Skew for the ", col, " variable: ", skew)
        # If skew is less than -1 or greater than 1, the distribution is highly skewed
        if skew < -1 or skew > 1:
            print("The distribution is highly skewed")
        # If skew is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed
        elif skew >= -1 and skew < -0.5 or skew > 0.5 and skew <= 1:
            print("The distribution is moderately skewed")

    # Study the normality of the variable
    if gaussian_test:
        # Do a qq-plot
        st.probplot(df[col], dist="norm", plot=plt)
        plt.show()
        # Perform the Shapiro-Wilk test for normality
        stat, p = st.shapiro(df[col])
        print('Statistics=%.4f, p=%.4f' % (stat, p))
        # Interpret the results
        if p > 0.05:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')

    outliers = df[(df[col] < lowerBound) | (df[col] > upperBound)]
    print("Number of outliers: ", len(outliers), " which is ", round( (len(outliers)*100)/len(df), 2), "% of the total dataset")

    return outliers

def study_column_discrete(df: pd.DataFrame, col: str, target=None, print_stats = True) -> None:
    """
    Performs a complete study of a column in a dataframe, for a discrete quantitative variable.
    - It prints the mean, median, standard deviation and mode of the column.
    - It plots a barplot to show the distribution of the column.

    Parameters
    -------
    df: pandas dataframe
        The dataframe containing the column to be studied.
    col: string
        The name of the column to be studied.
    target: string, optional
        The name of the target column for a classification problem. If provided, 
        it will plot a barplot with the target column as hue.
    
    Returns
    --------
    None
        A barplot or a countplot is plotted to show the distribution of the column.
    """
    if print_stats:
        print("The column ", col, " mean is ", df[col].mean())
        print("The column ", col, " median is ", df[col].median())
        print("The column ", col, " standard deviation is ", df[col].std())
        print("The column ", col, " mode is ", df[col].mode()[0])

    if target:
        # Plot the possible values for the 'col' column and the target variable
        sns.countplot(x=col, hue=target, data=df)
    else:
        sns.barplot(x=df[col].unique(), y=df[col].value_counts())
    plt.show()
    return

def study_column_categorical(df: pd.DataFrame, col: str, target=None):
    """
    Performs a complete study of a column in a dataframe, for a categorical variable.
    - It prints the mode and the number of unique values of the column.
    - It plots a pie chart to show the distribution of the different categories.
    - If a target variable is provided and it is not the column to be studied, it plots a barplot to show the distribution of the column, grouped by the target variable.

    Parameters
    -------
    df: pandas dataframe
        The dataframe containing the column to be studied.
    col: string
        The name of the column to be studied.
    target: string, optional
        The name of the target column for a classification problem. If provided, it will plot a barplot with the target column.

    Returns
    -------
    None
        A pie chart is plotted to show the distribution of the different categories.
    """

    print("The column ", col, " mode is ", df[col].mode()[0])
    print("The column ", col, " has ", len(df[col].unique()), " unique values")

    # Draw a piechart to show the distribution of occupations
    labels = df[col].value_counts().index
    sizes = df[col].value_counts().values
    # Explode will be used to highlight the largest slice
    explode = [0.1 if sizes[i] == max(sizes) else 0 for i in range(len(sizes))]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90) 
    ax1.axis('equal')
    plt.show()

    # Plot the possible values for the 'col' column and their frequency in percentage
    if target and target != col:
        sns.histplot(df, x=col, hue=target, stat="percent", multiple="dodge", shrink=.8)
    return

def count_outliers_per_column(df: pd.DataFrame):
    """
    Counts the number of outliers per each column in a dataframe.

    Parameters
    ----------
    df : pandas dataframe
        The dataframe to be studied.
    
    Returns
    -------
    outliers : pandas dataframe
        A dataframe with the number of outliers per column.
    """
    # Get the number of rows in the dataframe and create an empty dataframe to store the results
    len_df = len(df)
    outliers = pd.DataFrame(columns=["Column", "Number of high outliers", "Number of low outliers"])
    # Iterate over the numerical columns
    for col in df.select_dtypes(include=np.number).columns:
        # Calculate the number of high and low outliers for each column
        num_high_outliers = len(df[df[col] > df[col].quantile(0.75) + 1.5*(df[col].quantile(0.75) - df[col].quantile(0.25))])
        num_low_outliers = len(df[df[col] < df[col].quantile(0.25) - 1.5*(df[col].quantile(0.75) - df[col].quantile(0.25))])
        # Append the results to the outliers dataframe
        outliers = outliers.append({
            "Column": col, 
            "Number of High outliers": num_high_outliers, 
            "Number of Low outliers": num_low_outliers,
            "High outliers %": round((num_high_outliers*100)/len_df, 2),
            "Low outliers %": round((num_low_outliers*100)/len_df, 2)
        }, ignore_index=True)
    return outliers

def plot_outliers_per_column(df: pd.DataFrame, percent=False):
    """
    Plots a bar chart to show the number of outliers per column.

    Parameters
    ----------
    df : pandas dataframe
        The dataframe to be studied.
    percent : boolean, optional
        If True, the bar chart will show the percentage of outliers per column
        instead of the total count. The default is False.
    
    Returns
    -------
    None
        A bar chart is plotted.
    """
    # Get the number of outliers per column
    outliers = count_outliers_per_column(df)
    # Create a grid of 2 rows and 1 column
    fig, axs = plt.subplots(2, 1, figsize=(20,10))

    types = ["High", "Low"]
    for i in range(types):
        if percent:
            # Plot the percentage of outliers per column
            sns.barplot(x="Column", y=types[i]+" outliers %", data=outliers, ax=axs[i])
        else:
            # Plot the number of outliers per column
            sns.barplot(x="Column", y="Number of "+types[i]+" outliers", data=outliers, ax=axs[i])
    plt.show()
    return
    