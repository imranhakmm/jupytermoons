import sqlite3
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class Moons:
    def __init__(self, db_path):
        """
        Initializes the Moons class and loads data from an SQLite database.

        Args:
        db_path (str): Path to the SQLite database file.
        """
        # Establish a connection to the SQLite database
        self.conn = sqlite3.connect(db_path)
        
        # Load data into a pandas DataFrame
        self.data = pd.read_sql_query("SELECT * FROM moons", self.conn)
        
        # Number of moons in the dataset
        self.number_of_moons = "The number of moons is " + str(len(self.data))
        
        # Unique groups of moons in the dataset
        unique_groups = ', '.join(self.data['group'].unique())
        self.groups = "The 8 distinct groups of moons are " + unique_groups

    def update_moon_data(self):
        """
        Updates the DataFrame with new values for mass and magnitude.
        This method uses predefined values to update specific moons.
        """
        # Dictionary containing updated values for mass and magnitude from wikipedia
        updated_values = {
                'mass_kg': {
                     'Adrastea':2.00e+15,
                    'Aitne': 1.40e+13, 
                    'Amalthea':2.08e+18 ,
                    'Ananke':1.30e+16,
                    'Aoede': 3.40e+13,
                    'Arche':1.40e+13, 
                    'Autonoe':3.40e+13 ,
                    'Callirrhoe':4.60e+18,
                    'Callisto': 1.075938e+23,
                    'Carme':5.30e+16 ,
                    'Carpo':1.40e+13,
                    'Chaldene':3.40e+13, 
                    'Cyllene':4.20e+16,
                    'Dia':3.40e+13,
                    'Eirene':3.40e+13,
                    'Elara':2.70e+17,
                    'Erinome':1.40e+13 ,
                    'Ersa':1.40e+13,
                    'Euanthe':1.40e+13,
                    'Eukelade':3.40e+13,
                    'Eupheme':4.20e+12,
                    'Euporie':4.20e+12 ,
                    'Europa':4.7998e+22,
                    'Eurydome':1.40e+13,
                    'Ganymede':1.4819e+23,
                    'Harpalyke':3.40e+13 ,
                    'Hegemone':1.40e+13,
                    'Helike':3.40e+13 ,
                    'Hermippe':3.40e+13,
                    'Herse':4.20e+12,
                    'Himalia':4.20e+18,
                    'Io':8.931938e+22,
                    'Iocaste':6.50e+13,
                    'Isonoe':3.40e+13,
                    'Kale':4.20e+12 ,
                    'Kallichore':4.20e+12,
                    'Kalyke':1.70e+14,
                    'Kore':4.20e+12 ,
                    'Leda':5.20e+16,
                    'Lysithea':3.90e+16,
                    'Megaclite':6.50e+13,
                    'Metis':3.60e+16,
                    'Mneme':4.20e+12,
                    'Orthosie':4.20e+12,
                    'Pandia':1.40e+13,
                    'Pasiphae':1.00e+17,
                    'Pasithee':4.20e+12 ,
                    'Philophrosyne':4.20e+12,
                    'Praxidike':1.80e+14,
                    'Sinope':2.20e+16,
                    'Sponde':4.20e+12,
                    'Taygete':6.50e+13,
                    'Thebe':4.30e+17,
                    'Thelxinoe':4.20e+12,
                    'Themisto':3.80e+14,
                    'Thyone':3.40e+13,
                
                },
                'mag': {
                    'Adrastea': 19.1,  
                    'Metis': 11.9,
                    'Thebe' :16.9 ,
                }
            }

        # Iterate over the dictionary and update the DataFrame
        for column, updates in updated_values.items():
            for moon, value in updates.items():
                if pd.isna(self.data.loc[self.data['moon'] == moon, column]).any():
                    self.data.loc[self.data['moon'] == moon, column] = value

    def summary_statistics(self):
        """
        Returns summary statistics for the numerical columns in the dataset.

        Returns:
        DataFrame: Summary statistics of the numerical columns.
        """
        return self.data.describe()

    def plot_distribution(self):
        """
        Plots histograms for all numerical columns in the dataset in a grid layout.
        """
        # Identify numeric columns in the DataFrame
        numeric_columns = [col for col in self.data.columns if np.issubdtype(self.data[col].dtype, np.number)]

        # Handle the case when there are no numeric columns
        if not numeric_columns:
            print("No numeric columns to plot.")
            return

        # Calculate the layout dimensions for subplots
        n_cols = int(math.sqrt(len(numeric_columns))) or 1
        n_rows = int(math.ceil(len(numeric_columns) / n_cols))

        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), squeeze=False)

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        # Plot histograms for each numeric column
        for i, col in enumerate(numeric_columns):
            self.data[col].hist(ax=axes[i])
            axes[i].set_title(f'Histogram of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

        # Hide unused subplots
        for i in range(len(numeric_columns), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def filter_by_group(self, group_name):
        """
        Filters the dataset to return only the data for a specified moon group.

        Args:
        group_name (str): The name of the group to filter by.

        Returns:
        DataFrame: Filtered DataFrame containing only the data for the specified group.
        """
        return self.data[self.data['group'] == group_name]

    def plot_boxplot(self, y_column, x_column=None):
        """
        Creates a box plot for the specified column.
    
        Args:
        y_column (str): The name of the numerical column to be used for the box plot.
        x_column (str, optional): The name of the categorical column to group the data by.
        """
        if y_column in self.data.columns:
            sns.boxplot(data=self.data, x=x_column, y=y_column)
            plt.title(f'Box Plot of {y_column}' + (' by ' + x_column if x_column else ''))
            plt.xlabel(x_column if x_column else 'Index')
            plt.ylabel(y_column)
            plt.show()
        else:
            print(f"{y_column} is not a column in the dataset.")

    def plot_scatter(self, x_column, y_column, hue=None):
        """
        Creates a scatter plot for the specified columns.

        Args:
        x_column (str): The name of the column to be used as the x-axis.
        y_column (str): The name of the column to be used as the y-axis.
        hue (str, optional): The name of the column to be used for color encoding.
        """
        if x_column in self.data.columns and y_column in self.data.columns:
            sns.scatterplot(data=self.data, x=x_column, y=y_column, hue=hue)
            plt.title(f'Scatter Plot of {x_column} vs {y_column}')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.show()
        else:
            print(f"One or both of the specified columns: {x_column}, {y_column} are not in the dataset.")

    
    def plot_correlation_heatmap(self):
        """
        Creates a heatmap showing the correlations between numerical columns in the dataset.
        """
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()




class MassModelling(Moons):
    """
    A subclass of Moons for performing linear regression analysis 
    to estimate the mass of Jupiter using Kepler's Third Law.
    """

    def __init__(self, db_path):
        """
        Initialize the MassModelling class by calling the initialization of the Moons class.

        Args:
        db_path (str): Path to the SQLite database file.
        """
        super().__init__(db_path)  # Initialize the parent class

    def verify_linear_relationship(self):
        """
        Verifies the linear relationship between T^2 and a^3 by plotting a scatter plot 
        and calculating the Pearson correlation coefficient. This helps to assess the 
        suitability of linear regression for modeling this relationship.
        """
        # Prepare the data for analysis
        self.prepare_data()

        # Create a scatter plot to visually inspect the relationship
        sns.scatterplot(x=self.data['a_cubed'], y=self.data['T_squared'])
        plt.xlabel('Semi-major Axis Cubed (a^3) [m^3]')
        plt.ylabel('Orbital Period Squared (T^2) [s^2]')
        plt.title('Scatter Plot of T^2 vs a^3')
        plt.show()

        # Calculate and print the Pearson correlation coefficient
        correlation = self.data['T_squared'].corr(self.data['a_cubed'])
        print(f"Pearson correlation coefficient: {correlation}")

    def prepare_data(self):
        """
        Prepares the data by converting units and adding columns for T^2 and a^3.
        This is necessary to align the data with the formula used in Kepler's Third Law.
        """
        # Convert period from days to seconds (1 day = 86400 seconds)
        self.data['period_seconds'] = self.data['period_days'] * 86400

        # Convert distance from km to meters (1 km = 1000 meters)
        self.data['distance_meters'] = self.data['distance_km'] * 1000

        # Add columns for T^2 (period squared) and a^3 (semi-major axis cubed)
        self.data['T_squared'] = self.data['period_seconds'] ** 2
        self.data['a_cubed'] = self.data['distance_meters'] ** 3

    def train_regression_model(self):
        """
        Trains a linear regression model using the prepared data and plots the linear 
        regression lines for both the training and testing sets. This method provides
        insights into the model's fit and generalization capability.
        """
        # Prepare data for the regression model
        self.prepare_data()

        # Define the independent variable (a^3) and dependent variable (T^2)
        X = self.data[['a_cubed']]
        y = self.data['T_squared']

        # Split data into training and testing sets for model validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the linear regression model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        # Evaluate and print model performance on the test set
        y_pred_test = self.model.predict(X_test)
        y_pred_train = self.model.predict(X_train)
        r2 = r2_score(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        print(f"Model R-squared: {r2}, Mean Squared Error: {mse}")

        # Plotting the regression line for the training set
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train, y_train, color='blue', label='Training data')
        plt.plot(X_train, y_pred_train, color='green', label='Regression line - Train')
        plt.title('Linear Regression Model - Training Set')
        plt.xlabel('Semi-major Axis Cubed (a^3)')
        plt.ylabel('Orbital Period Squared (T^2)')
        plt.legend()
        plt.show()

        # Plotting the regression line for the testing set
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='red', label='Testing data')
        plt.plot(X_test, y_pred_test, color='orange', label='Regression line - Test')
        plt.title('Linear Regression Model - Testing Set')
        plt.xlabel('Semi-major Axis Cubed (a^3)')
        plt.ylabel('Orbital Period Squared (T^2)')
        plt.legend()
        plt.show()

    def calculate_and_compare_jupiter_mass(self):
        """
        Calculates the mass of Jupiter based on the linear regression model and compares it with 
        the known true mass. The method reports the estimated value, the discrepancy, and the 
        percentage discrepancy.
        """
        # Gravitational constant in m^3 kg^-1 s^-2
        G = 6.67430e-11  

        # Calculate the slope from the linear model (4*pi^2/GM)
        slope = self.model.coef_[0]

        # Rearrange to find M (mass of Jupiter)
        M_estimated = (4 * np.pi**2) / (G * slope)

        # Known true mass of Jupiter in kilograms
        M_true = 1.898e27  # in kg

        # Calculate the discrepancy
        discrepancy = abs(M_true - M_estimated)

        # Calculate the percentage discrepancy
        percentage_discrepancy = (discrepancy / M_true) * 100

        result = (
            f"Estimated Mass of Jupiter: {M_estimated:.4e} kg\n",
            f"True Mass of Jupiter: {M_true:.4e} kg\n",
            f"Discrepancy: {discrepancy:.4e} kg\n",
            f"Percentage Discrepancy: {percentage_discrepancy:.3f}%"
        )
        return result


        




  




