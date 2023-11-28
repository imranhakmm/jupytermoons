import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns


class Moons:
    def __init__(self, db_path):
        """
        Constructor for the Moons class. Loads the data from the SQLite database.
        """
        self.conn = sqlite3.connect(db_path)
        self.data = pd.read_sql_query("SELECT * FROM moons", self.conn)
        self.number_of_moons = len(self.data)
        self.groups = self.data['group'].unique()

    def summary_statistics(self):
        """
        Returns summary statistics for the dataset.
        """
        return self.data.describe()

    def plot_data(self, column_name, kind='hist'):
        """
        Generates a plot for a specified column.
        """
        if column_name not in self.data.columns:
            return f"Column '{column_name}' not found in the dataset."

        return self.data[column_name].plot(kind=kind, title=f"{column_name} Distribution")

    def get_moon_data(self, moon_name):
        """
        Returns data for a specified moon.
        """
        return self.data[self.data['moon'] == moon_name]

    def correlation_analysis(self):
        """
        Analyzes and returns the correlation matrix for the numeric columns in the dataset.
        """
        numeric_data = self.data.select_dtypes(include=[np.number])
        return numeric_data.corr()

    def prepare_kepler_data(self):
        """
        Prepares the data for Kepler's Third Law analysis.
        Converts T (period in days) to seconds and a (distance in km) to meters.
        Adds T^2 and a^3 columns to the data.
        """
        # Constants for unit conversion
        SECONDS_PER_DAY = 86400
        METERS_PER_KILOMETER = 1000

        # Converting period to seconds and distance to meters
        self.data['period_s'] = self.data['period_days'] * SECONDS_PER_DAY
        self.data['distance_m'] = self.data['distance_km'] * METERS_PER_KILOMETER

        # Calculating T^2 and a^3
        self.data['T_squared'] = self.data['period_s'] ** 2
        self.data['a_cubed'] = self.data['distance_m'] ** 3

    def setup_linear_model(self):
        """
        Initializes the linear regression model.
        """
        self.model = LinearRegression()

    def train_model(self):
        """
        Trains the linear regression model on the dataset.
        """
        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.data['a_cubed'].values.reshape(-1, 1),
                                                            self.data['T_squared'].values, test_size=0.2)

        # Training the model
        self.model.fit(X_train, y_train)

        # Testing the model
        self.test_score = self.model.score(X_test, y_test)

        return self.test_score

    def predict_mass_of_jupiter(self):
        """
        Uses Kepler's Third Law and the linear regression model to estimate the mass of Jupiter.
        """
        # Gravitational constant in m^3 kg^-1 s^-2
        G = 6.67430e-11

        # The slope of the linear regression model represents 4 * pi^2 / (G * Mass of Jupiter)
        slope = self.model.coef_[0]

        # Estimating the mass of Jupiter
        jupiter_mass = (4 * np.pi**2) / (G * slope)

        return jupiter_mass

    def compare_with_literature(self, literature_mass):
        """
        Compares the estimated mass of Jupiter with a known value from literature.
        """
        estimated_mass = self.predict_mass_of_jupiter()
        percent_diff = ((estimated_mass - literature_mass) / literature_mass) * 100

        return estimated_mass, percent_diff

class EnhancedMoons(Moons):
    def __init__(self, db_path):
        """
        Constructor for the EnhancedMoons class. Inherits from the Moons class.
        """
        super().__init__(db_path)

    def handle_missing_values(self):
        """
        Handles missing values in the dataset. Currently, it fills missing mass values with the mean.
        """
        self.data['mass_kg'].fillna(self.data['mass_kg'].mean(), inplace=True)

    def normalize_data(self):
        """
        Normalizes numeric columns in the dataset.
        """
        numeric_cols = self.data.select_dtypes(include=['float64']).columns
        self.data[numeric_cols] = (self.data[numeric_cols] - self.data[numeric_cols].mean()) / self.data[numeric_cols].std()

    def group_analysis(self):
        """
        Analyzes data based on moon groups, returning summary statistics for each group.
        """
        return self.data.groupby('group').describe()

    def enhanced_plotting(self, column_x, column_y):
        """
        Enhanced plotting method to visualize relationships between two variables.
        """
        if column_x not in self.data.columns or column_y not in self.data.columns:
            return f"One or both columns '{column_x}', '{column_y}' not found in the dataset."

        return self.data.plot(kind='scatter', x=column_x, y=column_y, title=f"{column_x} vs {column_y}")

    def export_data(self, file_name):
        """
        Exports the current state of the dataset to a CSV file.
        """
        self.data.to_csv(file_name, index=False)
        return f"Data exported to {file_name}"

    def data_quality_report(self):
        """
        Generates a report on data quality, including completeness and potential anomalies.
        """
        missing_values = self.data.isnull().sum()
        unique_values = self.data.nunique()
        return {"Missing Values": missing_values, "Unique Values": unique_values}

# Example Usage:
# moons = EnhancedMoons('path_to_jupiter.db')
# moons.handle_missing_values()
# moons.normalize_data()
# ...




