from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List

# Abstract Base Class for Bivariate Analysis Strategy
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform bivariate analysis on two features of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.

        Returns:
        None: This method visualizes the relationship between the two features.
        """
        pass


# Concrete Strategy for Numerical vs Numerical Analysis
# ------------------------------------------------------
# This strategy analyzes the relationship between two numerical features using scatter plots.
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two numerical features using a scatter plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first numerical feature/column to be analyzed.
        feature2 (str): The name of the second numerical feature/column to be analyzed.

        Returns:
        None: Displays a scatter plot showing the relationship between the two features.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
    
# Concrete Strategy for Numerical Features vs Target Variable
# ------------------------------------------------------
# This strategy analyzes the relationship between a list of numerical features and target variable using bar plots.
class NumericalFeaturesVsTargetAnalysis(BivariateAnalysisStrategy):    
    def analyze(self, df: pd.DataFrame, remove_cols: List[str], target: str):
        """
        Plots the relationship between the given numerical features and the target variable using bar plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        remove_cols (List[str]): The list of features of the dataframe which needs to be excluded
        target (str): The actual target variable of the dataframe.

        Returns:
        None: Displays a bar plot showing the relationship between the given numerical features and the target variable.
        """
        numerical_features = df.select_dtypes(include='number').columns.tolist()
        
        for col in remove_cols:
            if col in df.columns and col in numerical_features:
                numerical_features.remove(col)
            else:
                raise ValueError(f"Feature not present in the DataFrame: {col}")
            
        features = numerical_features 
        correlations = df[features + [target]].corr()[target].drop(target).sort_values()

        # Plot correlation heatmap
        plt.figure(figsize=(10, 6))
        sns.barplot(x=correlations.values, y=correlations.index, hue=correlations.index, palette='coolwarm', legend=False)
        plt.title('Correlation of Numerical Features with Efficiency')
        plt.xlabel('Correlation Coefficient')
        plt.tight_layout()
        plt.show()


# Concrete Strategy for Categorical vs Numerical Analysis
# --------------------------------------------------------
# This strategy analyzes the relationship between a categorical feature and a numerical feature using box plots.
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between a categorical feature and a numerical feature using a box plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the categorical feature/column to be analyzed.
        feature2 (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a box plot showing the relationship between the categorical and numerical features.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()


# Concrete Strategy for Categorical Features vs Target Variable
# ------------------------------------------------------
# This strategy analyzes the relationship between a list of categorical features and target variable using bar plots.
class CategoricalFeaturesVsTargetAnalysis(BivariateAnalysisStrategy):    
    def analyze(self, df: pd.DataFrame, remove_cols: List[str], target: str):
        """
        Plots the relationship between the given categorical features and the target variable using barplot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        remove_cols (List[str]): The list of features of the dataframe which needs to be excluded
        target (str): The actual target variable of the dataframe.

        Returns:
        None: Displays a bar plot showing the relationship between the given categorical features and the target variable.
        """
        categorical_features = df.select_dtypes(include='object').columns.tolist()
        print(categorical_features)

        for col in remove_cols:
            if col in df.columns and col in categorical_features:
                categorical_features.remove(col)
            else:
                raise ValueError(f"Feature not present in the DataFrame: {col}")
            
        features = categorical_features 
        for feature in features:
            if df[feature].nunique() > 50:
                print(f"Skipping {feature} — too many unique values.")
                continue

            plt.figure(figsize=(10, 5))
            sns.barplot(data=df, x=feature, y=target, hue=feature, legend=False, errorbar='sd', palette='viridis')
            plt.title(f'{feature} vs {target}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


# Concrete Strategy for Categorical vs Categorical Analysis
# --------------------------------------------------------
# This strategy analyzes the relationship between two categorical features using heatmap and stacked bar plot.
class CategoricalVsCategoricalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two categorical features using a heatmap and stacked bar plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first categorical feature/column to be analyzed.
        feature2 (str): The name of the second categorical feature/column to be analyzed.

        Returns:
        None: Displays a heatmap and stacked bar plot showing the relationship between the two categorical features.
        """
        # Create a contingency table
        contingency_table = pd.crosstab(df[feature1], df[feature2], normalize='index')
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot heatmap
        sns.heatmap(contingency_table, annot=True, fmt='.2%', cmap='YlOrRd', ax=ax1)
        ax1.set_title(f'Heatmap of {feature1} vs {feature2}')
        ax1.set_xlabel(feature2)
        ax1.set_ylabel(feature1)
        
        # Plot stacked bar chart
        contingency_table.plot(kind='bar', stacked=True, ax=ax2)
        ax2.set_title(f'Stacked Bar Chart of {feature1} vs {feature2}')
        ax2.set_xlabel(feature1)
        ax2.set_ylabel('Proportion')
        ax2.legend(title=feature2, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        # Print chi-square test results
        from scipy.stats import chi2_contingency
        chi2, p_value, dof, expected = chi2_contingency(pd.crosstab(df[feature1], df[feature2]))
        print(f"\nChi-square test results:")
        print(f"Chi-square statistic: {chi2:.2f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Degrees of freedom: {dof}")
        print("\nInterpretation:")
        if p_value < 0.05:
            print("There is a significant relationship between the two categorical variables (p < 0.05)")
        else:
            print("There is no significant relationship between the two categorical variables (p >= 0.05)")

# Context Class that uses a BivariateAnalysisStrategy
# ---------------------------------------------------
# This class allows you to switch between different bivariate analysis strategies.
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the BivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (BivariateAnalysisStrategy): The strategy to be used for bivariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new strategy for the BivariateAnalyzer.

        Parameters:
        strategy (BivariateAnalysisStrategy): The new strategy to be used for bivariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str = None, feature2: str = None, remove_cols: List[str] = None, target: str = None):
        """
        Executes the bivariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str, optional): The name of the first feature/column to be analyzed.
        feature2 (str, optional): The name of the second feature/column to be analyzed.
        remove_cols (List[str], optional): The list of features to exclude.
        target (str, optional): The target variable name.

        Returns:
        None: Executes the strategy's analysis method and visualizes the results.
        """
        if remove_cols is not None and target is not None:
            self._strategy.analyze(df, remove_cols, target)
        else:
            self._strategy.analyze(df, feature1, feature2)
    

# Example usage
if __name__ == "__main__":
    # # Example usage of the BivariateAnalyzer with different strategies.

    # # Load the data
    # df = pd.read_csv('Dataset/Extracted/train.csv')

    # # Analyzing relationship between two categorical features
    # analyzer = BivariateAnalyzer(CategoricalVsCategoricalAnalysis())
    # analyzer.execute_analysis(df, 'installation_type', 'manufacturer')

    # # Analyzing relationship between two numerical features
    # analyzer = BivariateAnalyzer(CategoricalFeaturesVsTargetAnalysis())
    # analyzer.execute_analysis(df, [], 'efficiency')

    # # Analyzing numerical features vs target
    # analyzer = BivariateAnalyzer(NumericalFeaturesVsTargetAnalysis())
    # analyzer.execute_analysis(df, [], 'efficiency')

    # # Analyzing relationship between a categorical and a numerical feature
    # analyzer.set_strategy(CategoricalVsNumericalAnalysis())
    # analyzer.execute_analysis(df, 'installation_type', 'efficiency')
    pass
