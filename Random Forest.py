import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  # Import the new tool used to beat the baseline
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np
from scipy.stats import wilcoxon  # Import statistical test tool

def main():
    # System settings (4 main systems).
    systems = ['xz', 'h2', 'dconvert', 'x264'] 
    num_repeats = 30  # Meet strict experimental requirements
    train_frac = 0.7  
    random_seed = 1 

    for current_system in systems:
        datasets_location = 'datasets/{}'.format(current_system)  # Modify this to specify the location of the datasets

        # Check if path exists
        if not os.path.exists(datasets_location):
            print(f"Path not found: {datasets_location}, please confirm if the folder location is correct!")
            continue

        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')] 

        for csv_file in csv_files:
            print(f'\n======================================================')
            print(f'> System: {current_system}, Dataset: {csv_file}')
            data = pd.read_csv(os.path.join(datasets_location, csv_file))

            # Store test results for baseline model (LR) and proposed model (RF) separately
            metrics_lr = {'MAPE': [], 'MAE': [], 'RMSE': []}
            metrics_rf = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                # Randomly split data
                train_data = data.sample(frac=train_frac, random_state=random_seed*current_repeat) 
                test_data = data.drop(train_data.index)

                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                # 1. Baseline Model: Linear Regression
                lr_model = LinearRegression() 
                lr_model.fit(training_X, training_Y) 
                lr_preds = lr_model.predict(testing_X) 

                metrics_lr['MAPE'].append(mean_absolute_percentage_error(testing_Y, lr_preds))
                metrics_lr['MAE'].append(mean_absolute_error(testing_Y, lr_preds))
                metrics_lr['RMSE'].append(np.sqrt(mean_squared_error(testing_Y, lr_preds)))

                # 2. New tool: Random Forest
                # n_estimators=100 means using 100 trees, random_state=42 ensures reproducibility
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(training_X, training_Y)
                rf_preds = rf_model.predict(testing_X)

                metrics_rf['MAPE'].append(mean_absolute_percentage_error(testing_Y, rf_preds))
                metrics_rf['MAE'].append(mean_absolute_error(testing_Y, rf_preds))
                metrics_rf['RMSE'].append(np.sqrt(mean_squared_error(testing_Y, rf_preds)))

            # Output and compare results
            lr_avg_mape = np.mean(metrics_lr['MAPE'])
            rf_avg_mape = np.mean(metrics_rf['MAPE'])
            
            print(f"【Average MAPE】 Baseline(LR): {lr_avg_mape:.4f} | New tool(RF): {rf_avg_mape:.4f}")
            print(f"【Average MAE】 Baseline(LR): {np.mean(metrics_lr['MAE']):.2f} | New tool(RF): {np.mean(metrics_rf['MAE']):.2f}")
            print(f"【Average RMSE】 Baseline(LR): {np.mean(metrics_lr['RMSE']):.2f} | New tool(RF): {np.mean(metrics_rf['RMSE']):.2f}")

            # Perform Statistical Test (Wilcoxon Signed-Rank Test)
            # Use MAPE, the key metric for the test
            stat, p_value = wilcoxon(metrics_lr['MAPE'], metrics_rf['MAPE'])
            print(f"【 p-value】: {p_value:.4e}")
            
            if p_value < 0.05 and rf_avg_mape < lr_avg_mape:
                print("Conclusion: Random Forest is statistically significantly better than the Baseline model!")
            elif p_value < 0.05 and rf_avg_mape > lr_avg_mape:
                print("Conclusion: Baseline model is better; Random Forest performed poorly on this dataset.")
            else:
                print("Conclusion: Difference is not statistically significant (p >= 0.05).")

if __name__ == "__main__":
    main()