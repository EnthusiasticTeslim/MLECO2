import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def to_np(y):
    '''Converts torch tensor to numpy array
    params:
        y: torch tensor
    returns:
        y: numpy array
    '''
    return y.cpu().detach().numpy()

def cu_fraction(Sn):
    '''Calculates the fraction of Cu in the catalyst
    params:
        Sn: float, the percentage of Sn in the catalyst
    returns:
        Cu: float, the percentage of Cu in the catalyst
    '''
    if Sn <= 1:
        Cu = 1 - Sn
    else:
        raise ValueError('Sn percent must be less than or equal to 1')
    return Cu

def get_weight(Sn):
    '''Calculates the weight of the catalyst
    params:
        Sn: float, the percentage of Sn in the catalyst
    returns:
        weight: float, the weight of the catalyst
    '''
    # create the structure
    if Sn <= 1:
        weight = (1 - Sn)*63.546 + (Sn)*118.71
    else:
        raise ValueError('Sn percent must be less than or equal to 1')
    return weight

def preprocessing(df: np.array):
    '''Preprocess the data
    params:
        df: numpy array, the input data
    returns:
    '''
    df = torch.from_numpy(df).float()
    return df

def load_data(data_path):
    '''Load the data and preprocess it  '''
    # the data
    data = pd.read_excel(data_path)


    # drop the S/N column
    if 'S/N' in data.columns:
        data = data.drop(columns=['S/N'])

    # normalize the data in target columns by 100
    features_col = list(data.columns[:4])
    target_col = list(data.columns[4:])

    data[target_col] = data[target_col] / 100 # normalize the target data by 100
    data[features_col[2]] = data[features_col[2]] / 100 # normalize the Sn% by 100
    print('Features: ', features_col)
    print('Target: ', target_col)

    data['Cu %'] = 1 - data['Sn %']
    data['weight'] = data['Sn %'].apply(get_weight)

    # normalize the data in features columns to range [0, 1]
    features_col += ['Cu %', 'weight']
    print(f'New features: {features_col}')
    minX = data[features_col].min()
    maxX = data[features_col].max()

    data[features_col] = (data[features_col] - minX) / (maxX - minX)

    return data, features_col, target_col


def plot_heat_map(
                    data: pd.DataFrame = None, mask: bool = False, compute_corr: bool = True,
                    fig_size = (10, 5), save_fig: bool = False, name: str = 'general'
                    ):
    ''' Plot the heatmap of the correlation matrix of the data  '''
        
    fig, ax = plt.subplots(1, figsize=fig_size, facecolor='white')

    # Create the heatmap for the original data
    if compute_corr:
        corr = data.corr(method='pearson')
    else:
        corr = data
    if mask is False:
        sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', linewidths=.5, ax=ax, cbar=False)
    else: # mask the diagonal
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', linewidths=.5, ax=ax, cbar=False, mask=mask)
        
    # Show the plot
    plt.show()
    if save_fig:
        fig.savefig(f'./reports/heatmap_{name}.png', dpi=200)




def mse(y_true, y_pred, torch=False):
    '''Calculate the mean squared error'''
    if torch:
        y_true = y_true.detach().numpy()
        y_pred = y_pred.detach().numpy()
    return round(float(mean_squared_error(y_true, y_pred)), 3)

def mae(y_true, y_pred, torch=False):
    '''Calculate the mean absolute error'''
    if torch:
        y_true = y_true.detach().numpy()
        y_pred = y_pred.detach().numpy()
    return round(float(mean_absolute_error(y_true, y_pred)), 3)

def r2(y_true, y_pred, torch=False):
    '''Calculate the R2 score'''
    if torch:
        y_true = y_true.detach().numpy()
        y_pred = y_pred.detach().numpy()
    return round(float(r2_score(y_true, y_pred)), 3)

def calculate_metrics(y_true, y_pred, torch=True):
    r2_score = r2(y_true, y_pred, torch)
    mae_score = mae(y_true, y_pred, torch)
    rmse_score = mse(y_true, y_pred, torch)
    return r2_score, mae_score, rmse_score

def calculate_and_write_metrics(y_train, y_test, output_train_list, output_test_list, plot_target_column, result_dir):
    with open(f'{result_dir}/metrics.txt', 'w') as f:
        for column_j, column_name in enumerate(plot_target_column):
            f.write(f"Target {column_name}\n")
            
            train_metrics = []
            test_metrics = []
            
            for model_i, (output_train, output_test) in enumerate(zip(output_train_list, output_test_list)):
                train_metrics.append(calculate_metrics(y_train[:, column_j], output_train[:, column_j]))
                test_metrics.append(calculate_metrics(y_test[:, column_j], output_test[:, column_j]))
                
                f.write(f"Model {model_i}\n")
                f.write(f"Train R2: {train_metrics[-1][0]:.3f}, Test R2: {test_metrics[-1][0]:.3f}, "
                        f"Train MAE: {train_metrics[-1][1]:.3f}, Test MAE: {test_metrics[-1][1]:.3f}, "
                        f"Train RMSE: {train_metrics[-1][2]:.3f}, Test RMSE: {test_metrics[-1][2]:.3f}\n")
            
            train_metrics = np.array(train_metrics)
            test_metrics = np.array(test_metrics)
            
            f.write("Average\n")
            f.write(f"Train R2: {np.mean(train_metrics[:, 0]):.3f} +/- {np.std(train_metrics[:, 0]):.3f}, "
                    f"Test R2: {np.mean(test_metrics[:, 0]):.3f} +/- {np.std(test_metrics[:, 0]):.3f}, "
                    f"Train MAE: {np.mean(train_metrics[:, 1]):.3f} +/- {np.std(train_metrics[:, 1]):.3f}, "
                    f"Test MAE: {np.mean(test_metrics[:, 1]):.3f} +/- {np.std(test_metrics[:, 1]):.3f}, "
                    f"Train RMSE: {np.mean(train_metrics[:, 2]):.3f} +/- {np.std(train_metrics[:, 2]):.3f}, "
                    f"Test RMSE: {np.mean(test_metrics[:, 2]):.3f} +/- {np.std(test_metrics[:, 2]):.3f}\n")
            f.write("\n")