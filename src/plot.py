import sys
sys.dont_write_bytecode = True

from .utils import to_np
import numpy as np
import seaborn as sns

# Scatter plot function
def plot_scatter(ax, data, row, col, column_names, torch=True):
    if torch:
        ax[row, col].scatter(to_np(data['y_train'][:, col]), to_np(data['output_train'][:, col]), color='blue', label='Train')
        ax[row, col].scatter(to_np(data['y_test'][:, col]), to_np(data['output_test'][:, col]), color='green', label='Test')
    else:
        ax[row, col].scatter(data['y_train'][:, col], data['output_train'][:, col], color='blue', label='Train')
        ax[row, col].scatter(data['y_test'][:, col], data['output_test'][:, col], color='green', label='Test')

    ax[row, col].plot([0, 1], [0, 1], transform=ax[row, col].transAxes, color='red', linestyle='--')
    ax[row, col].set_title(column_names[col])
    ax[row, col].set_xlabel(r'$\rm True\ FE$')
    if row == 0 and col == 0:
        ax[row, col].legend()
        ax[row, col].set_ylabel(r'$\rm Predicted\ FE$')
        

# KDE plot function
def plot_kde(ax, data, row, col, torch=True):
    if torch:
        y_data = np.concatenate([to_np(data['y_train'][:, col]), to_np(data['y_test'][:, col])])
        output_data = np.concatenate([to_np(data['output_train'][:, col]), to_np(data['output_test'][:, col])])
    else:
        y_data = np.concatenate([data['y_train'][:, col], data['y_test'][:, col]])
        output_data = np.concatenate([data['output_train'][:, col], data['output_test'][:, col]])
        
    error = np.abs(y_data - output_data)
    _, _, hist = ax[row, col].hist(error, density=True, alpha=0.5)
    sns.kdeplot(error, ax=ax[row, col], color=hist.patches[0].get_facecolor(), lw=2)
    ax[row, col].set_xlabel(r'$\rm Error$')
    
    # if row == 1 and col == 0:
    #     ax[row, col].set_ylabel(r'$\rm Density$')

import matplotlib.pyplot as plt
import numpy as np

def plot_result(
                output_train_list: list, output_test_list: list, 
                y_train: list, y_test: list, 
                target_column: list, 
                index: int = 0, 
                max_value: float = 1, normalised: bool = False,
                figsize: tuple = (5, 4)):
    
    plt.figure(figsize=figsize)

    def process_data(data_list, y_data):
        data = np.array([to_np(output[:, index]) for output in data_list])
        avg = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return y_data[:, index], avg, std

    y_train_plot, train_avg, train_std = process_data(output_train_list, y_train)
    y_test_plot, test_avg, test_std = process_data(output_test_list, y_test)

    plt.errorbar(y_train_plot, train_avg, yerr=train_std, fmt='o', capsize=5, capthick=2, 
                 ecolor='blue', color='blue', label=r'$\rm {Train}$')
    plt.errorbar(y_test_plot, test_avg, yerr=test_std, fmt='o', capsize=5, capthick=2, 
                 ecolor='green', color='green', label=r'$\rm {Test}$')
    if normalised:
        plt.plot([0, max_value], [0, max_value], 'k--', label=r'$\rm y=x$')
    else:
        plt.plot([0, max_value*100], [0, max_value*100], 'k--', label=r'$\rm y=x$')

    plt.xlabel(fr'$\rm {{Exp.\ {target_column[index]}}}$', fontsize=15)
    plt.ylabel(fr'$\rm {{ML.\ {target_column[index]}}}$', fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()