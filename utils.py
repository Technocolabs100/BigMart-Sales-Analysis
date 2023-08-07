import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def sliceDict(table, pairCount):
    return dict(itertools.islice(table.items(), pairCount))

def info(df):
    print(f"<class 'pandas.core.frame.DataFrame'>")
    print(f"RangeIndex: {len(df)} entries, 0 to {len(df) - 1}")
    print(f"Data columns (total {len(df.columns)} columns):")
    print(" # \tColumn\t\t\t\tNull_Count  Dtype\tUnique_Count")
    print("---\t------\t\t\t\t----------  -----\t------------")
    dtype_count = {}
    
    for i, col in enumerate(df.columns):
        null_count = df[col].isnull().sum()
        dtype = df[col].dtype
        unique_count = len(df[col].unique())
            
        if dtype in dtype_count:
            dtype_count[dtype] += 1
        else:
            dtype_count[dtype] = 1
        
        print(f"{i}\t{col.ljust(25)}\t{str(null_count).ljust(12)}{dtype}\t{unique_count}")
    print(f"dtypes: {', '.join([f'{dtype}({count})' for dtype, count in dtype_count.items()])}")
    print(f"memory usage: {df.memory_usage().sum() / 1024:.1f}+ KB")
    return None


def compare_hist(original, scaled):
    fig, ax = plt.subplots(1, 2, figsize=(15, 3))
    sns.histplot(original, ax=ax[0], kde=True, legend=False)
    ax[0].set_title("Original data")
    sns.histplot(scaled, ax=ax[1], kde=True, legend=False)
    ax[1].set_title("Scaled data")
    return None


def visualizeNumericalFeatures(dataFrame, numericalFeatures, needsScaling=True, scaler=MinMaxScaler()):
    x = dataFrame[numericalFeatures].copy()
    # Apply scaling to the numerical columns
    x[numericalFeatures] = scaler.fit_transform(x[numericalFeatures])
    sns.kdeplot(x)
    plt.title('Distribution of Numerical Features')
    #plt.legend(loc='best')
    plt.show()
    print()
    print("Statistical Overview of Numerical Features:\n")
    return dataFrame.describe()


def visualizeFrequency(dataFrame, categoricalFeatures, height=5.5, width=15, style='bar'):
    # Grid Size
    cols = 3
    rows = (len(categoricalFeatures) + cols - 1) // cols
    plt.figure(figsize=(width, height * rows))
    for i, feature in enumerate(categoricalFeatures):
        plt.subplot(rows, cols, i+1)  # Create a subplot for each column
        if style == 'bar':
            sns.countplot(data=dataFrame, x=feature, palette='tab20')
            plt.ylabel(''); plt.xlabel('')
            plt.title(feature)
            if dataFrame[feature].nunique() > 3:
                plt.xticks(rotation=90)
        elif style == 'donut':
            # Get value counts for the feature
            value_counts = dataFrame[feature].value_counts()
            labels = value_counts.index.tolist()
            sizes = value_counts.values.tolist()
            # Define colors for the donut chart
            colors = sns.color_palette('tab20', n_colors=len(categoricalFeatures))
            # Create a donut chart
            wedges, _, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', wedgeprops=dict(width=0.4, edgecolor='w'), colors=colors)
            # Enhance text properties for percentage labels
            plt.gca().add_artist(plt.Circle((0,0),0.7,fc='white'))
            plt.axis('equal')
            plt.title(feature)
    plt.tight_layout()
    plt.suptitle("Frequency of Categorical Features")
    plt.subplots_adjust(top=0.9)
    plt.show()
    print()
    print("Statistical Overview of Categorical Features:\n")
    return dataFrame.describe(include='object')


# Functions for bivariate analysis

def count_unique_values(dataframe, B, A):
    """
    Print the count of unique values in column B for each unique value in column A.
    Parameters:
        * dataframe (pd.DataFrame): The DataFrame containing the data.
        * column_B (str): The name of the column B.
        * column_A (str): The name of the column A.
    """
    unique_Bs_for_each_A = dataframe.groupby(A)[B].nunique().reset_index() 
    print(unique_Bs_for_each_A.rename(columns={B: ("nunique " + B)}))
    return unique_Bs_for_each_A


def getDistributionByCategory(dataframe, categoricalFeature, numericalFeature, plot=True):
    # Group the data by "categoricalFeature" and calculate the sum of numericalFeature.
    distribution = dataframe.groupby(categoricalFeature)[numericalFeature].sum()
    distribution = distribution.sort_values(ascending=False)
    if plot:
        sns.lineplot(y=distribution.index, x=distribution.values)
        plt.ylabel(categoricalFeature)
        plt.xlabel(f"Total {numericalFeature}")
        plt.title(categoricalFeature)
    return distribution

