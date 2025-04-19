import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


class TitanicEDA:
    """
    A class to perform exploratory data analysis on the Titanic dataset.
    """

    def __init__(self, data_path=None):
        """
        Args : data_path (str, optional) : path to the Titanic dataset CSV file
        If not provided, will attempt to download the dataset.
        """

        self.data_path = data_path
        self.data = None
        self.output_dir = "visualization"

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.load_data()

    def load_data(self):
        """
        Load the Titanic dataset either from the provided path or download it.

        Returns : pandas.DataFrame : The loaded Titanic dataset
        """

        if self.data_path and os.path.exists(self.data_path):
            self.data = pd.read_csv(self.data_path)
        else:
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

            try:
                self.data = pd.read_csv(url)
                print(f"Data download successfully from {url}")
            except Exception as e:
                print(f"Error downloading data : {e}, Might network is down !!!")
                return None
            
            self.data["Age"].fillna(self.data["Age"].median(), inplace=True)
            self.data["Embarked"].fillna(self.data["Embarked"].mode()[0], inplace=True)
            self.data["Fare"].fillna(self.data["Fare"].median(), inplace=True)

            return self.data

    def get_summary_stats(self):
        """
        Generate summary statistics for the Titanic dataset.

        Returns : 
            dict : A dictionary containing various summary statistics
        """
        if self.data is None:
            print("Data Not Loaded")
            return None

        stats = {
            "total_passengers" : len(self.data),
            "survived_count" : self.data["Survived"].sum(),
            "survial_rate" : self.data["Survived"].mean() * 100,
            "gender_distribution" : self.data["Sex"].value_counts().to_dict(),
            "class_distribution" : self.data["Pclass"].value_counts().to_dict(),
            "age_stats" : {
                "mean":self.data["Age"].mean(),
                "median":self.data["Age"].median(),
                "min":self.data["Age"].min(),
                "max":self.data["Age"].max()
            },
            "numerical_summary":self.data.describe().to_dict()
        }

        return stats

    def print_summary(self):
        """
        Print a formatted summary of the dataset.
        """
        if self.data is None:
            print("Data not loaded. Call load_data() first.")
            return
        
        stats = self.get_summary_stats()
        
        print(f"===== Titanic Dataset Summary =====")
        print(f"Total passengers: {stats['total_passengers']}")
        print(f"Survived: {stats['survived_count']} ({stats['survial_rate']:.2f}%)")
        print(f"\nGender distribution:")
        for gender, count in stats['gender_distribution'].items():
            print(f"  - {gender}: {count}")
        
        print(f"\nPassenger class distribution:")
        for pclass, count in stats['class_distribution'].items():
            print(f"  - Class {pclass}: {count}")
        
        print(f"\nAge statistics:")
        print(f"  - Average age: {stats['age_stats']['mean']:.2f} years")
        print(f"  - Median age: {stats['age_stats']['median']:.2f} years")
        print(f"  - Age range: {stats['age_stats']['min']:.1f} to {stats['age_stats']['max']:.1f} years")
        
        # Print first few rows of the data
        print("\nSample data:")
        print(self.data.head())


    def visualize_survival_by_features(self, feature, save=True):
        """
        Visualize survival rates by a specific feature.

        Args:
            feature (str) : The feature to analyze (e.g., "Sex", "Pclass", etc.)
            save (bool) : Whether to save the visualization as an image file.

        Returns:
            matplotlib.figure.Figure : The genrated figure
        """

        if self.data is None:
            print("Data not loaded, Might be there is no network.")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))

        if feature in self.data.columns:

            survial_by_feature = self.data.groupby([feature])["Survived"].mean().sort_values()

            ax = survial_by_feature.plot(kind="bar", color = "skyblue")
            plt.title(f"Survival rate by {feature}")
            plt.ylabel("Survival Rate")
            plt.xlabel(feature)
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            for i, v in enumerate(survial_by_feature):
                ax.text(i, v + 0.02, f'{v:.2%}', ha='center')

            plt.tight_layout()

            if save:
                filename = f"survial_by_{feature.lower()}.png"
                filepath = os.path.join(self.output_dir,  filename)
                plt.savefig(filepath)
                print(f"Figure saved as {filepath}")

            return fig
        
        else:
            print(f"Feature '{feature}' not found in the dataset")
            return None

    def visualize_age_distribution(self, save=True):
        """
        Visualize the age distribution between survivors and non-survivors.

        Args:
            save (bool) : Whether to save the visualization as an image file.
        
        Returns:
            matplotlib.figure.Figure : The genrated figure
        """

        if self.data is None:
            print("Data not loaded.")
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        survived = self.data[self.data["Survived"] == 1]["Age"]
        died = self.data[self.data["Survived"] == 0]["Age"]

        ax.hist([survived, died], bins=30, stacked=False, alpha=0.7, label=["Survived", "Did not survived"])

        plt.title("Age, Distribution by survival status")
        plt.xlabel("Age")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(axis="y", linestyle='--', alpha=0.7)

        if save:
            filepath = os.path.join(self.output_dir, "age_distribution.png")
            plt.savefig(filepath)
            print(f"Figure saved as {filepath}")

        return fig


    def visualize_correlation_matrix(self, save=True):
        """
        Visualize the correlation matrix between numerical features.

        Args:
            save (bool) : Whether to save the visualization as an image file.

        Returns:
            matplotlib.figure.Figure : The genrated figure
        """

        if self.data is None:
            print("Data not loaded.")
            return None

        df_corr = self.data.copy()

        le = LabelEncoder()
        for col in df_corr.select_dtypes(include=["object"]).columns:
            df_corr[col] = le.fit_transform(df_corr[col].astype(str))

        numeric_data = df_corr.select_dtypes(include=["int64", "float64"])
        corr_matrix = numeric_data.corr()


        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        plt.title("Correlation Matrix of Features")
        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, "corr_mat.png")
            plt.savefig(filepath)
            print(f"Figure saved as {filepath}")

        return fig
            

    def run_all_visualizations(self):
        """
        Run all visualization methods and save the results.
        """

        if self.data is None:
            print("Data is not loaded.")
            return 
        
        self.visualize_survival_by_features("Sex")
        self.visualize_survival_by_features("Pclass")
        self.visualize_survival_by_features("Embarked")

        self.data["AgeBin"] = pd.cut(self.data["Age"], bins=[0, 12, 18, 35, 60, 100],
                                     labels = ["Child", "Teenager", "Young Adult", "Adult", "Senior"])
        

        self.visualize_survival_by_features('Agebin')

        self.visualize_age_distribution()
        self.visualize_correlation_matrix()

        print(f"All visualizations have been genrated and saved to the '{self.output_dir}' directory.")

if __name__ == "__main__":
    eda = TitanicEDA()

    eda.load_data()

    eda.print_summary()

    eda.run_all_visualizations()


    