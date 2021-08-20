# -*- coding: utf-8 -*-
"""
@author: VictorGueorguiev
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

def main():
    data_path = '..\\.\\output\\OUT\\PLOT\\'
    models = {'pfh': 'Perceptron FH',
              'pcms': 'Perceptron PCMS',
              'nbfh': 'Naive Bayes FH',
              'nbcms': 'Naive Bayes NBCMS',
              'adpfh': 'Enchanced Perceptron FH'}
    PLOT_COLORS = ['C'+str(i) for i in range(1, 20)]

    metric_names = ['acc', 'recall', 'precision', 'balancedaccuracy'] # possible are acc, f1score, recall, precision, balancedaccuracy 
    plot_metric_names = ['Accuracy', 'Recall', 'Precision', 'Balanced Accuracy']
    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()
    i = 0
    for metric_name in metric_names:
        ax = axs[i]
        plot_color = 0
        for file_name in glob.glob(data_path + '*.' + metric_name):
            model_name = file_name.split(data_path)[1].split('.')[1]
            model_name = models[model_name]
            plot_df = pd.read_csv(file_name, sep = '\t')
            plot_df.columns = ['training_samples', 'metric_name']
            
            ax.plot(plot_df.training_samples, 
                    plot_df.metric_name, 
                    color = PLOT_COLORS[plot_color],
                    #marker = 'o',
                    linestyle = '-',
                    label = model_name)
            plot_color += 1
        ax.set_xlabel('No. Training Samples')
        ax.set_ylabel(plot_metric_names[i])
        ax.legend(loc="lower right")
        i += 1
    plt.legend()
    plt.show()

        

    
    
    
if __name__ == "__main__":
    main()

