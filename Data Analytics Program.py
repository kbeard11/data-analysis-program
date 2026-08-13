#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from IPython.display import display, Markdown


# Data Analysis Program
# =====================

# This program provides a full CLI workflow for:
    # - Inspecting a dataset
    # - Computing summary statistics
    # - Creating EDA visualisations
    # - Training and evaluating ML regression models

# The user navigates through menus to perform
# different analysis tasks interactively


# Load dataset from CSV file
# ==========================
# Returns a DataFrame if successful, otherwise
# prints an error message and returns None

def read_dataset():
    """
    Load the dataset from a CSV file and return it as a pandas DataFrame.

    Returns
    -------
    DataFrame or None
        The loaded dataset if found, otherwise None with an error message.
    """
    try:
        df = pd.read_csv("C:/Documents/data.csv")
        return df

    except FileNotFound:
        print("File not found: File has been moved or renamed")
        return None

# Dataset Inspection Module
# =========================

# Allows the user to:
    # 1. View first 5 rows
    # 2. View last 5 rows
    # 3. View dataset summary 

# This helps users understand the structure and quality of the dataset before analysis


def inspect_df():
    """
    Interactive menu for inspecting the dataset.

    Allows the user to:
    - View the first 5 rows
    - View the last 5 rows
    - Review dataset structure (df.info)

    This function does not return anything; it prints results directly.
    """
    while True:
        print("\n------------------------------")
        print("-----Data Inspection Menu-----")
        print("------------------------------")
        print("\nPlease Select From The Following Data Inspection Options:\n")
        print("1. Show First 5 Rows Of The Dataset")
        print("2. Show Last 5 Rows Of The Dataset")
        print("3. Review A Summary Of The Dataset")
        print("4. Return To Main Menu\n") 

        inspect_option = input("Enter 1-4: ")

        if inspect_option == "1":
            print("\nHere Are The First 5 Rows Of the Dataset\n")
            inspect_head = df.head()
            print(inspect_head)

        elif inspect_option == "2":
            print("\nHere Are The Last 5 Rows Of the Dataset\n")
            inspect_tail = df.tail()
            print(inspect_tail)

        elif inspect_option == "3":
            print("\nHere Is A Summary Of The Dataset\n")
            inspect_info = df.info()
            print(inspect_info)

        elif inspect_option == "4":
            break

        else:
            print("Invalid Input: Please Enter 1-4")


# Summary Statistics Module
# =========================

# Displays descriptive statistics (mean, std, min, max, etc.) of the dataset
# User chooses whether to view the summary

def stats_module(df):
    """
    Display descriptive summary statistics for the dataset.

    Parameters
    ----------
    df : DataFrame
        The dataset to analyse.

    Notes
    -----
    Uses df.describe() to show mean, std, min, max, and quartiles.
    """
    summary_stats = df.describe()

    user_input = input("Display Summary Statistics of Dataset?\n")

    if user_input == "Yes":
        print(summary_stats)

    elif user_input == "No":
        print("Exiting Summary Statistics")
        main()

    else:
        print("Invalid Input: Please input Yes or No to proceed")


# Variable Statistics Module
# ==========================

# Computes mean, median, and standard deviation for selected variables (age, height, weight)
# User selects which statistic to compute

def age_module(df):
    """
    Compute summary statistics for the 'age' variable.

    Parameters
    ----------
    df : DataFrame
        The dataset containing the 'age' column.

    Notes
    -----
    Allows the user to select mean, median, or standard deviation.
    """
    mean_stats_age = df["age"].mean()
    median_stats_age = df["age"].median()
    std_stats_age = df["age"].std()

    user_input = input("Select a statistics calculation for age. Mean = 1, Median = 2, Standard Deviation = 3: \n")

    if user_input == "1":
        print(mean_stats_age)

    elif user_input == "2":
        print(median_stats_age)

    elif user_input == "3":
        print(std_stats_age)

    else:
        print("Invalid Input: Please Enter 1-3")

def height_module(df):  
    """
    Compute summary statistics for the 'height' variable.

    Parameters
    ----------
    df : DataFrame
        The dataset containing the 'height' column.

    Notes
    -----
    Allows the user to select mean, median, or standard deviation.
    """
    mean_stats_height = df["height"].mean()
    median_stats_height = df["height"].median()
    std_stats_height = df["height"].std()

    user_input = input("Select a statistics calculation for height. Mean = 1, Median = 2, Standard Deviation = 3: \n")

    if user_input == "1":
        print(mean_stats_height)

    elif user_input == "2":
        print(median_stats_height)

    elif user_input == "3":
        print(std_stats_height)

    else:
        print("Invalid Input: Please Enter 1-3")

def weight_module(df):
    """
    Compute summary statistics for the 'weight' variable.

    Parameters
    ----------
    df : DataFrame
        The dataset containing the 'weight' column.

    Notes
    -----
    Allows the user to select mean, median, or standard deviation.
    """
    mean_stats_weight = df["weight"].mean()
    median_stats_weight = df["weight"].median()
    std_stats_weight = df["weight"].std()

    user_input = input("Select a statistics calculation for weight. Mean = 1, Median = 2, Standard Deviation = 3: \n")

    if user_input == "1":

        print(mean_stats_weight)

    elif user_input == "2":
        print(median_stats_weight)

    elif user_input == "3":
        print(std_stats_weight)

    else:
        print("Invalid Input: Please Enter 1-3")


# Exploratory Data Analysis (EDA)
# ===============================

# Provides histogram plots and simple linear trend visualisations between variables
# Helps users understand distributions and relationships

def eda_vis_module(df):
    """
    Interactive menu for exploratory data analysis (EDA).

    Provides:
    - Histograms for age, height, and weight
    - Scatter plots showing relationships between variables
    - Linear trend line visualisations using numpy.polyfit

    Parameters
    ----------
    df : DataFrame
        The dataset used for visualisation.
    """
    while True:
        print("\n----------------------------")
        print("-----Visualisation Menu-----")
        print("----------------------------")
        print("\nSelect From the Following Visualisation Options: \n")
        print("1. Age Histogram")
        print("2. Height Histogram")
        print("3. Weight Histogram")
        print("4. Relationship Between Age and Height")
        print("5. Relationship Between Age and Weight")
        print("6. Relationship Between Height and Weight")
        print("7. Return To Main Menu\n")

        option = input("Enter 1-7: ")

        # Functions are called based on user input to the main menu.
        if option == "1":
            age_vis = df["age"].plot(kind="hist", bins=8)
            plt.xlabel("Age")
            plt.ylabel("Frequency")
            plt.title("Distribution Of Age")
            plt.show()

        elif option == "2": 
            height_vis = df["height"].plot(kind="hist", bins=8)
            plt.xlabel("Height")
            plt.ylabel("Frequency")
            plt.title("Distribution Of Height")
            plt.show()

        elif option == "3":
            weight_vis = df["weight"].plot(kind="hist", bins=8)
            plt.xlabel("Weight")
            plt.ylabel("Frequency")
            plt.title("Distribution Of Weight")
            plt.show()

        elif option == "4":
            age_per_height = df.groupby("age")["height"].mean()
            x = age_per_height.index
            y = age_per_height.values
            m, b = np.polyfit(x,y,1)
            print(f"y = {b:.3f}+{m:.3f}x")
            plt.scatter(x,y)
            plt.plot(x,m*x + b, color="red")
            plt.xlabel("Age")
            plt.ylabel("Average Height")
            plt.title("Average Height Per Age")
            plt.show()

        elif option == "5":
            age_per_weight = df.groupby("age")["weight"].mean()
            x = age_per_weight.index
            y = age_per_weight.values
            m, b = np.polyfit(x,y,1)
            print(f"y = {b:.3f}+{m:.3f}x")
            plt.scatter(x,y)
            plt.plot(x,m*x + b, color="red")
            plt.xlabel("Age")
            plt.ylabel("Average Weight")
            plt.title("Average Weight Per Age")
            plt.show()

        elif option == "6":
            height_per_weight = df.groupby("height")["weight"].mean()
            x = height_per_weight.index
            y = height_per_weight.values
            m, b = np.polyfit(x,y,1)
            print(f"y = {b:.3f}+{m:.3f}x")
            plt.scatter(x,y)
            plt.plot(x,m*x + b, color="red")
            plt.xlabel("Weight")
            plt.ylabel("Height")
            plt.title("Average Weight Per Height")
            plt.show()

        elif option == "7":
            print("Exiting Menu")
            break
        else:
            print("Invalid Input: Please Enter 1-7")


# Machine Learning Module
# =======================

# Workflow:
    # 1. Optional correlation matrix heatmap
    # 2. User selects target variable (height or weight)
    # 3. Train/test split
    # 4. StandardScaler normalisation
    # 5. User selects ML model:
        # - Linear Regression
        # - Support Vector Regression
        # - Random Forest Regression
        # - Decision Tree Regression
    # 6. Model is trained and evaluated using:
        # - R2
        # - MSE
        # - RMSE
        # - MAE
    # 7. Optional bar‑chart comparison of training/testing metrics

def machine_module(df):
    """
    Machine learning workflow menu for regression modelling.

    Workflow includes:
    - Optional correlation matrix heatmap
    - Target variable selection (height or weight)
    - Train/test split
    - Feature scaling using StandardScaler
    - Model selection:
        * Linear Regression
        * Support Vector Regression
        * Random Forest Regression
        * Decision Tree Regression
    - Training and testing evaluation using R2, MSE, RMSE, MAE
    - Optional bar-chart comparison of model performance

    Parameters
    ----------
    df : DataFrame
        The dataset used for training and evaluation.
    """
    while True:
        print("\n----------------------------")
        print("-----Correlation Matrix-----")
        print("----------------------------")
        print("\nWould You Like To Analyse The Correlation Between Variables? \n")
        print("1. Yes")
        print("2. No")
        print("3. Return To Main Menu\n")

        if corrmatrix_option == "1":
            plt.figure(figsize = (14,10))
            sns.heatmap(df.corr(), annot=True, cmap = 'coolwarm')
            plt.show()

        elif corrmatrix_option == "2":
            continue

        elif corrmatrix_option == "3":
            break

        else:
            print("Invalid Input: Please Enter 1-3")

        print("\n---------------------------------")
        print("-----Target Variable Selection-----")
        print("-----------------------------------")
        print("\nWhich Variable Would You Like To Predict? \n")
        print("1. Height")
        print("2. Weight")
        print("3. Return To Main Menu\n")

        target = input("Enter 1-3: ")

        if target == "1":
            X = df[["age", "weight"]]
            y = df["height"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
            print("\nYou have chosen to predict Height using Age and Weight\n")

        elif target == "2":
            X = df[["age", "height"]]
            y = df["weight"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
            print("\nYou have chosen to predict Weight using Age and Height\n")

        elif option == "3":
            break

        else:
            print("Invalid Input: Please Enter 1-3")

        # Normalise data using scalar transformation
        scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)

        print("\n-------------------------------")
        print("-----Machine Learning Menu-----")
        print("-------------------------------")
        print("\nSelect From the Following: \n")
        print("1. Show Correlation Matrix")
        print("2. Perform Linear Regression")
        print("3. Perform Support Vector Regression")
        print("4. Perform Random Forest Regression")
        print("5. Perform Decision Tree Regression")
        print("6. Exit Menu\n")

        ml_option = input("Enter 1-5: ")

        if ml_option == "1":
            model_lr = LinearRegression()
            model_lr.fit(scaled_X_train, y_train)
            train_pred_model_lr = model_lr.predict(scaled_X_train)
            y_pred_model_lr = model_lr.predict(scaled_X_test)

            train_r2_model_lr = r2_score(y_train, train_pred_model_lr)
            train_mse_model_lr = mean_squared_error(y_train, train_pred_model_lr)
            train_rmse_model_lr = np.sqrt(train_mse_model_lr)
            train_mae_model_lr = mean_absolute_error(y_train, train_pred_model_lr)
            training_results = {"Linear Regression": [train_r2_model_lr, train_mse_model_lr, train_rmse_model_lr, train_mae_model_lr]}
            print(f"\nLinear Regression Training:\nR2: {train_r2_model_lr}\nMSE: {train_mse_model_lr}\nRMSE: {train_rmse_model_lr}\nMAE: {train_mae_model_lr}\n")

            r2_model_lr = r2_score(y_test, y_pred_model_lr)
            mse_model_lr = mean_squared_error(y_test, y_pred_model_lr)
            rmse_model_lr = np.sqrt(mse_model_lr)
            mae_model_lr = mean_absolute_error(y_test, y_pred_model_lr)
            testing_results = {"Linear Regression": [r2_model_lr, mse_model_lr, rmse_model_lr, mae_model_lr]}
            print(f"Linear Regression Testing:\nR2: {r2_model_lr}\nMSE: {mse_model_lr}\nRMSE: {rmse_model_lr}\nMAE: {mae_model_lr}\n")


        elif ml_option == "2": 
            model_svm = SVR()
            model_svm.fit(scaled_X_train, y_train)
            train_pred_model_svm = model_svm.predict(scaled_X_train)
            y_pred_model_svm = model_svm.predict(scaled_X_test)

            train_r2_model_svm = r2_score(y_train, train_pred_model_svm)
            train_mse_model_svm = mean_squared_error(y_train, train_pred_model_svm)
            train_rmse_model_svm = np.sqrt(train_mse_model_svm)
            train_mae_model_svm =  mean_absolute_error(y_train, train_pred_model_svm)
            training_results = {"Support Vector Regression": [train_r2_model_svm, train_mse_model_svm, train_rmse_model_svm, train_mae_model_svm]}
            print(f"\nSupport Vector Regression Training:\nr2: {train_r2_model_svm}\nMSE: {train_mse_model_svm}\nRMSE: {train_rmse_model_svm}\nMAE:{train_mae_model_svm}\n")

            r2_model_svm = r2_score(y_test, y_pred_model_svm)
            mse_model_svm = mean_squared_error(y_test, y_pred_model_svm)
            rmse_model_svm = np.sqrt(mse_model_svm)
            mae_model_svm = mean_absolute_error(y_test, y_pred_model_svm)
            testing_results = { "Support Vector Regression": [r2_model_svm, mse_model_svm, rmse_model_svm, mae_model_svm]}
            print(f"Support Vector Regression Testing:\nr2: {r2_model_svm}\nMSE: {mse_model_svm}\nRMSE: {rmse_model_svm}\nMAE: {mae_model_svm}\n")

        elif ml_option == "3":
            model_rf = RandomForestRegressor()
            model_rf.fit(scaled_X_train, y_train)
            train_pred_model_rf = model_rf.predict(scaled_X_train)
            y_pred_model_rf = model_rf.predict(scaled_X_test)

            train_r2_model_rf = r2_score(y_train, train_pred_model_rf)
            train_mse_model_rf = mean_squared_error(y_train, train_pred_model_rf)
            train_rmse_model_rf = np.sqrt(train_mse_model_rf)
            train_mae_model_rf =  mean_absolute_error(y_train, train_pred_model_rf)
            training_results = {"Random Forest Regression": [train_r2_model_rf, train_mse_model_rf, train_rmse_model_rf, train_mae_model_rf]}
            print(f"\nRandom Forest Regressor Training:\nr2: {train_r2_model_rf}\nMSE: {train_mse_model_rf}\nRMSE: {train_rmse_model_rf}\nMAE:{train_mae_model_rf}\n")

            r2_model_rf = r2_score(y_test, y_pred_model_rf)
            mse_model_rf = mean_squared_error(y_test, y_pred_model_rf)
            rmse_model_rf = np.sqrt(mse_model_rf)
            mae_model_rf = mean_absolute_error(y_test, y_pred_model_rf)
            testing_results = {"Random Forest Regression": [r2_model_rf, mse_model_rf, rmse_model_rf, mae_model_rf]}
            print(f"Random Forest Regressor Testing:\nr2: {r2_model_rf}\nMSE: {mse_model_rf}\nRMSE: {rmse_model_rf}\nMAE: {mae_model_rf}\n")

        elif ml_option == "4":
            model_dt = DecisionTreeRegressor()
            model_dt.fit(scaled_X_train, y_train)
            train_pred_model_dt = model_dt.predict(scaled_X_train)
            y_pred_model_dt = model_dt.predict(scaled_X_test)

            train_r2_model_dt = r2_score(y_train, train_pred_model_dt)
            train_mse_model_dt = mean_squared_error(y_train, train_pred_model_dt)
            train_rmse_model_dt = np.sqrt(train_mse_model_dt)
            train_mae_model_dt =  mean_absolute_error(y_train, train_pred_model_dt)
            training_results = {"Decision Tree Regression": [train_r2_model_dt, train_mse_model_dt, train_rmse_model_dt, train_mae_model_dt]}
            print(f"\nDecision Tree Regressor Training:\nr2: {train_r2_model_dt}\nMSE: {train_mse_model_dt}\nRMSE: {train_rmse_model_dt}\nMAE:{train_mae_model_dt}\n")

            r2_model_dt = r2_score(y_test, y_pred_model_dt)
            mse_model_dt = mean_squared_error(y_test, y_pred_model_dt)
            rmse_model_dt = np.sqrt(mse_model_dt)
            mae_model_dt = mean_absolute_error(y_test, y_pred_model_dt)
            testing_results = {"Decision Tree Regression": [r2_model_dt, mse_model_dt, rmse_model_dt, mae_model_dt]}
            print(f"Decision Tree Regressor Testing:\nr2: {r2_model_dt}\nMSE: {mse_model_dt}\nRMSE: {rmse_model_dt}\nMAE: {mae_model_dt}\n")

        elif ml_option == "5":
            print("Exiting Menu")
            break

        else:
            print("Invalid Input: Input must be an option between 1 and 6")

        model_traintest_results = [training_results, testing_results]

        print("\n----------------------------------")
        print("-----Model Performance Review-----")
        print("----------------------------------")
        print("\nWould You Like To Graph The Training/Testing Performance Results Of Your Chosen Model? \n")
        print("Yes")
        print("No\n")

        results_option = input("Enter Yes or No: ")

        if results_option.casefold() == "yes":

            model_traintest_results[0] = pd.DataFrame(model_traintest_results[0], index=["r2", "MSE", "RMSE", "MAE"]).T
            metric_colors = {"r2": "red", "MSE": "green", "RMSE": "blue", "MAE": "orange"}

            fig, axes = plt.subplots(1, 2, figsize=(14,6))
            model_traintest_results[0].plot(kind="bar", ax=axes[0], color = metric_colors)
            axes[0].set_title("Machine Learning Model Training Results")
            axes[0].set_xlabel("\nMachine Learning Model")
            axes[0].set_ylabel("Training Score")
            for label in axes[0].get_xticklabels():
                label.set_rotation(0)

            model_traintest_results[1] = pd.DataFrame(model_traintest_results[1], index=["r2", "MSE", "RMSE", "MAE"]).T
            metric_colors = {"r2": "red", "MSE": "green", "RMSE": "blue", "MAE": "orange"}

            model_traintest_results[1].plot(kind="bar", ax=axes[1], color = metric_colors)
            axes[1].set_title("Machine Learning Model Testing Results")
            axes[1].set_xlabel("\nMachine Learning Model")
            axes[1].set_ylabel("Testing Score")
            for label in axes[1].get_xticklabels():
                label.set_rotation(0)
            plt.show()
            break

        elif results_option.casefold() == "no":
            break

        else:
            ("Invalid Input: Please Enter Yes or No")


# Main Program Menu
# =================

# Central navigation hub for the entire program
# Routes user to:
# - Dataset inspection
# - Summary statistics
# - Visualisations
# - Machine learning module
# - Exit program

def main():
    """
    Main program loop providing the user interface.

    Routes the user to:
    - Dataset inspection
    - Summary statistics
    - Visualisation module
    - Machine learning module
    - Program exit

    This function controls the overall application flow.
    """
    while True:
        display(Markdown("###### ========================================="))
        display(Markdown("#### -----Data Analysis Program Main Menu-----"))
        display(Markdown("###### ========================================="))
        print("\nSelect From the Following Options: \n")
        print("1. Inspect Dataset")
        print("2. Summary Statistics")
        print("3. Visualisation")
        print("4. Machine Learning")
        print("5. Exit Program\n")

        option = input("Enter 1-5: ")

        if option == "1": 
            inspect_df()

        elif option == "2":
            stats_module(df)

        elif option == "3":
            eda_vis_module(df)

        elif option == "4":
            machine_module(df)

        elif option == "5":
            print("\nYou Have Successfully Exited The Program")
            break

        else:
            print("Invalid Input: Please Enter 1-5")

if __name__ == "__main__":
    df = read_dataset()
    main()


# In[ ]:




