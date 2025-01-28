import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN


# Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# Class to handle data visualization tasks
class DataVisualizer:
    def __init__(self, data, output_folder="output_visuals"):
        self.data = data
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def save_plot(self, plot_name):
        """Check if the plot already exists, and save it if not."""
        plot_path = os.path.join(self.output_folder, plot_name)
        if not os.path.exists(plot_path):
            plt.savefig(plot_path)
            print(f"Plot saved as {plot_name}")
        else:
            print(f"{plot_name} already exists, skipping saving.")

    def plot_stacked_channels(self):
        """Plot visual representation of different channels when stacked independently."""
        plot_name = "stacked_channels.png"
        plot_path = os.path.join(self.output_folder, plot_name)
        
        # Check if the plot already exists, if it does, skip plotting
        if os.path.exists(plot_path):
            print(f"{plot_name} already exists, skipping plot creation.")
            return
        
        fig, axs = plt.subplots(5, sharex=True, sharey=True)
        fig.set_size_inches(18, 24)
        labels = ["X20", "X40", "X60", "X80", "X100"]
        colors = ["b", "g", "k", "r", "y"]
        fig.suptitle('Visual representation of different channels when stacked independently', fontsize=20)

        for i, ax in enumerate(axs):
            axs[i].plot(self.data.iloc[:, 0], self.data[labels[i]], color=colors[i], label=labels[i])
            axs[i].legend(loc="upper right")

        plt.xlabel('Total number of observations', fontsize=20)
        x_ticks = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
        x_ticklabels = ['0', '1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000', '12000']
        plt.xticks(x_ticks, x_ticklabels)

        # Save or skip saving the figure
        self.save_plot(plot_name)
        plt.show()

    def plot_stacked_against_each_other(self):
        """Plot visual representation of different channels when stacked against each other."""
        plot_name = "stacked_against_each_other.png"
        plot_path = os.path.join(self.output_folder, plot_name)
        
        # Check if the plot already exists, if it does, skip plotting
        if os.path.exists(plot_path):
            print(f"{plot_name} already exists, skipping plot creation.")
            return
        
        plt.rcParams["figure.figsize"] = (20, 10)
        self.data.loc[:, ::25].plot()
        plt.title("Visual representation of different channels when stacked against each other")
        plt.xlabel("Total number of values of x")
        plt.ylabel("Range of values of y")

        # Save or skip saving the figure
        self.save_plot(plot_name)
        plt.show()

    def plot_heatmap(self):
        """Plot heatmap for the correlation matrix."""
        plot_name = "heatmap.png"
        plot_path = os.path.join(self.output_folder, plot_name)
        
        # Check if the plot already exists, if it does, skip plotting
        if os.path.exists(plot_path):
            print(f"{plot_name} already exists, skipping plot creation.")
            return
        
        data_1 = self.data.copy()
        data_1.drop(['Unnamed', 'y'], axis=1, inplace=True)
        corr = data_1.corr()
        ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap='coolwarm')
        plt.title("Heat Map")

        # Save or skip saving the figure
        self.save_plot(plot_name)
        plt.show()

# Class to handle data preprocessing tasks
class DataProcessor:
    def __init__(self, data):
        self.data = data

    def check_for_null_values(self):
        """Check for missing values in the dataset."""
        null_values = self.data.isnull().sum()
        print(null_values.to_numpy())

    def handle_imbalance(self):
        """Handle imbalance in the dataset."""
        print(Counter(self.data['y']))

        # Apply SMOTE to solve class imbalance (example)
        data_2 = self.data.drop(["Unnamed"], axis=1).copy() 
        data_2["Output"] = data_2.y == 0
        data_2["Output"] = data_2["Output"].astype(int)
        data_2.y.value_counts() 
        data_2['y'] = data_2['y'].replace([2, 3, 4, 5], 0)
        data_2.y.value_counts()  # We can see there is a major class imbalance problem in our dataset
        
        X = data_2.drop(['Output', 'y'], axis=1)
        y = data_2['y']

        counter = Counter(y)
        print('Before SMOTE:', counter)

        # Oversampling the train dataset using SMOTE + ENN
        smenn = SMOTEENN()
        X_train1, y_train1 = smenn.fit_resample(X, y)

        counter = Counter(y_train1)
        print('After SMOTE:', counter)

        # Save the resampled dataset
        resampled_data = pd.DataFrame(X_train1, columns=X.columns)
        resampled_data['y'] = y_train1
        
        # Save the processed dataset
        resampled_data.to_csv('processed_data.csv', index=False)
        print("Resampled dataset saved as 'processed_data.csv'.")




# Main script
if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv('Epileptic Seizure Recognition.csv')

    # Create data visualizer and processor objects
    visualizer = DataVisualizer(data)
    processor = DataProcessor(data)

    # Check for missing values
    processor.check_for_null_values()

    # Handle class imbalance (SMOTE example)
    processor.handle_imbalance()

    # Visualize data
    visualizer.plot_stacked_channels()
    visualizer.plot_stacked_against_each_other()
    visualizer.plot_heatmap()
