import numpy as np
import matplotlib.pyplot as plt
from utils.data_manipulators import fill_max


def forward_fill(input_arr, max_len=100):
#     print(input_arr.shape)
    if max_len is None:
        max_len = float('-inf')
        for value in input_arr:
            if max_len < len(value):
                max_len = len(value)
    
    padded_arr = np.zeros((len(input_arr), max_len, len(input_arr[0][0])))
    
    for i, value in enumerate(input_arr):
        value = np.array(value)
        if len(value) < max_len:
            padded_arr[i,:len(value), :] = value[:len(value), :]
            padded_arr[i, len(value):, :] = padded_arr[i,len(value)-1, :]
        else:
            padded_arr[i,:max_len, :] = value[:max_len, :]

    return padded_arr

def plot_multiple_learning_plots(coefficients, path):

    for i in range(10):
        c = np.array(coefficients[i])
        plt.figure()
        
        plt.plot(c[:,0]/c.sum(axis=1), '#aa0a0a', label='Onemax')
        plt.plot(c[:,1]/c.sum(axis=1), '#a1aa21', label='Onemin')
        if c.shape[1] == 3:
            plt.plot(c[:,2]/c.sum(axis=1), '#a90ee0', label='Target')
        
        if c.shape[1] == 2:
            design_plot(wct=False, y_label='p')
        else:
            design_plot(wct=False, y_label='c')

        plt.savefig(path.format(i))




def plot_mean_std(n_run_values, color, label, shape, 
                  x_time=None, std=True, shape_interval=5,
                  do_fill_max=False, data_std=None, x_interval=1):
    mean = n_run_values.mean(0)
    if do_fill_max:
        mean = fill_max(mean)

    if data_std is None:
        ub = mean + n_run_values.std(0)
        lb = mean - n_run_values.std(0)
    else:
        ub = mean + data_std
        lb = mean - data_std
 
    if x_time is None:
        x_time = range(0, mean.shape[0]*x_interval, x_interval)
    
    if std:    
        plt.fill_between(x_time, ub, lb,
                     color=color, alpha=.1)
        
    markers_on_range = list(range(0, mean.shape[0]))
    if len(x_time)%shape_interval!=1:
        markers_on = list(np.append(markers_on_range[::shape_interval], markers_on_range[-1]))
    else:
        markers_on = markers_on_range[::shape_interval]
    
    plt.plot(x_time, mean, shape, color=color, label=label, markevery=markers_on)
    
def design_plot(x_label='w', y_label='a', loc='best'):
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    if x_label=='w':
        plt.xlabel('Wall Clock Time', fontsize=13.5)
    elif x_label=='g':
        plt.xlabel('Number of Generations', fontsize=13.5)
    else:
        plt.xlabel('Number of Function Evaluations', fontsize=13.5)
    
    if y_label=='a':
        plt.ylabel('Averaged Objective Value', fontsize=13.5)
    elif y_label=='m':
        plt.ylabel('Maximized Objective Value', fontsize=13.5)
    elif y_label=='c':
        plt.ylabel(r"Transfer Coefficients ($w_s$`$s$)", fontsize=13.5)
    elif y_label=='p':
        plt.ylabel('Probability', fontsize=13.5)
    plt.legend(prop={'size': 11})
        

def save_plot(addr):
#    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
#                 hspace = 0, wspace = 0)
   plt.savefig(addr, bbox_inches='tight', pad_inches = 0)

def plot_time_bars(execution_times, corresponding_labels, target_time, out_addr):
    
    relative_et = np.array(execution_times)
    
    # True Positives
    for i, t in enumerate(execution_times):
            
        plt.bar(i,t, color=SRC_COLORS[i], label=corresponding_labels[i])
#             plt.text(j+i-get_text_margin(results[i][k]), results[i][k] + 5, str(results[i][k]))

    axes = plt.gca()
    
    axes.spines['right'].set_visible(False)
#     axes.spines['left'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.set_yticks([])

    axes.set_facecolor((0, 0, 0, 0.0))
    axes.set_ylim(0, 1.1*np.max(relative_et))
    axes.set_xticks(list(range(len(relative_et))))
#     axes.set_xlim(-1, 13)
    axes.set_xticklabels(corresponding_labels)
#     plt.legend()
    
#     plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
#     plt.margins(0,0)
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.ylabel('Relative elapsed time to AMCTEA')
    plt.savefig(out_addr, bbox_inches='tight', pad_inches = 0)
