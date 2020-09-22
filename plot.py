import os
import re
import glob
import numpy as np
import matplotlib.pylab as plt
import matplotlib

from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from itertools import chain, count
from collections import defaultdict

from plot_util import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

method_labels_map = { 
    'MFH_Sampling':     'FH',
    'FH_Sampling':      'FH$^-$',
    'NH_Sampling':      'NH',
    'NH_LCCS_Sampling': 'NH',
    'MFH':              'FH-wo-S',
    'FH':               'FH$^-$-wo-S',
    'NH':               'NH-wo-S', 
    'NH_LCCS':          'NH-wo-S', 
    'EH':               'EH',
    'BH':               'BH', 
    'MH':               'MH',
    'Random_Scan':      'Random-Scan',
    'Sorted_Scan':      'Sorted-Scan',
    'Linear':           'Linear-Scan'
}

dataset_labels_map = {
    'MovieLens150': 'MovieLens',
    'Netflix300':   'Netflix',
    'Yahoo300':     'Yahoo',
    'Mnist':        'Mnist',
    'Sift':         'Sift',
    'Gaussian':     'Gaussian',
    'Gist':         'Gist',
    'Yelp':         'Yelp',
    'Music':        'Music-100',
    'GloVe100':     'GloVe',
    'Tiny1M':       'Tiny-1M',
    'Msong':        'Msong'
}

# datasets = ['Yelp', 'GloVe100']
datasets = ['Yelp', 'Music', 'GloVe100', 'Tiny1M', 'Msong']
dataset_labels = [dataset_labels_map[dataset] for dataset in datasets]

method_colors  = [
    'red', 'blue', 'green', 
    'purple', 'deepskyblue', 'darkorange', 
    'olive',  'deeppink', 'dodgerblue', 'dimgray']
method_markers = ['o', '^', 's', 'd', '*', 'p', 'x', 'v', 'D', '>']


# ------------------------------------------------------------------------------
def calc_width_and_height(n_datasets, n_rows):
    '''
    calc the width and height of figure

    :params n_datasets: number of dataset (integer)
    :params n_rows: number of rows (integer)
    :returns: width and height of figure
    '''
    fig_width  = 0.55 + 3.333 * n_datasets
    fig_height = 0.8 + 2.7 * n_rows
    
    return fig_width, fig_height


# ------------------------------------------------------------------------------
def get_filename(input_folder, dataset_name, method_name):
    '''
    get the file prefix 'dataset_method'

    :params input_folder: input folder (string)
    :params dataset_name: name of dataset (string)
    :params method_name:  name of method (string)
    :returns: file prefix (string)
    '''
    name = '%s%s_%s.out' % (input_folder, dataset_name, method_name)
    return name


# ------------------------------------------------------------------------------
def parse_res(filename, chosen_top_k):
    """
    parse result and get info such as ratio, qtime, recall, index_size, 
    chosen_k, and the setting of different methods
    
    alsh_ml: m=20, l=2, 
    Indexing Time: 4.588209 Seconds
    Estimated Memory: 42.868721 MB
    nc=20, cand=10
    1	16.307552	0.040309	26.200001	1.310000
    5	11.153884	0.042216	25.480000	5.308332
    10	8.998750	0.042983	25.600000	8.827541
    15	7.465528	0.044214	25.966648	11.455925
    20	6.482063	0.045658	26.280001	13.476919

    nc=20, cand=100
    1	16.307552	0.039867	26.200001	1.310000
    5	11.153884	0.041901	25.480000	5.308332
    10	8.998750	0.042529	25.600000	8.827541
    15	7.465528	0.044109	25.966648	11.455925
    20	6.482063	0.044947	26.280001	13.476919
    """
    setting_pattern = re.compile(r'\S+\s+.*=.*')

    setting_m = re.compile(r'.*(m)=(\d+).*')  # MFH, FH, NH, NH_LCCS BH, MH
    setting_l = re.compile(r'.*(l)=(\d+).*')  # MFH, FH, NH, BH, MH
    setting_M = re.compile(r'.*(M)=(\d+).*')  # only MH
    setting_s = re.compile(r'.*(s)=(\d+).*')  # MFH, FH, NH, NH_LCCS with sampling
    setting_b = re.compile(r'.*(b)=(\d+\.\d+).*')  # MFH, MH, BH with multi-partition
    
    param_settings = [setting_m, setting_l, setting_M, setting_s, setting_b]

    index_time_pattern   = re.compile(r'Indexing Time: (\d+\.\d+).*')
    memory_usage_pattern = re.compile(r'Estimated Memory: (\d+\.\d+).*')
    candidate_pattern    = re.compile(r'.*cand=(\d+).*')
    records_pattern      = re.compile(r'(\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)')

    params = {}
    with open(filename, 'r') as f:
        for line in f:
            res = setting_pattern.match(line)
            if res:
                for param_setting in param_settings:
                    tmp_res = param_setting.match(line)
                    if tmp_res is not None:
                        # print(tmp_res.groups())
                        params[tmp_res.group(1)] = tmp_res.group(2)
                # print("setting=", line)

            res = index_time_pattern.match(line)
            if res:
                chosen_k = float(res.group(1))
                # print('chosen_k=', chosen_k)
            
            res = memory_usage_pattern.match(line)
            if res:
                memory_usage = float(res.group(1))
                # print('memory_usage=', memory_usage)

            res = candidate_pattern.match(line)
            if res:
                cand = int(res.group(1))
                # print('cand=', cand)
            
            res = records_pattern.match(line)
            if res:
                top_k     = int(res.group(1))
                ratio     = float(res.group(2))
                qtime     = float(res.group(3))
                recall    = float(res.group(4))
                precision = float(res.group(5))
                fraction  = float(res.group(6))
                # print(top_k, ratio, qtime, recall, precision, fraction)

                if top_k == chosen_top_k:
                    yield ((cand, params), (top_k, chosen_k, memory_usage, 
                        ratio, qtime, recall, precision, fraction))


# ------------------------------------------------------------------------------
def getindexingtime(res):
    return res[1]
def getindexsize(res):
    return res[2]
def getratio(res):
    return res[3]
def gettime(res):
    return res[4]
def getrecall(res):
    return res[5]
def getprecision(res):
    return res[6]
def getfraction(res):
    return res[7]

def get_cand(res):
    return int(res[0][0])
def get_l(res):
    return int(res[0][1]['l'])
def get_m(res):
    return int(res[0][1]['m'])
def get_s(res):
    return int(res[0][1]['s'])
def get_time(res):
    return float(res[1][4])
def get_recall(res):
    return float(res[1][5])
def get_precision(res):
    return float(res[1][6])
def get_fraction(res):
    return float(res[1][7])


# ------------------------------------------------------------------------------
def lower_bound_curve(xys):
    '''
    get the time-recall curve by convex hull and interpolation

    :params xys: 2-dim array (np.array)
    :returns: time-recall curve with interpolation
    '''
    # add noise and conduct convex hull to find the curve
    eps = np.random.normal(size=xys.shape) * 1e-6
    xys += eps
    # print(xys)
    
    hull = ConvexHull(xys)
    hull_vs = xys[hull.vertices]
    # hull_vs = np.array(sorted(hull_vs, key=lambda x:x[1]))
    # print("hull_vs: ", hull_vs)

    # find max pair (maxv0) and min pairs (v1s) from the convex hull
    v1s = []
    maxv0 = [-1, -1]
    for v0, v1 in zip(hull_vs, chain(hull_vs[1:], hull_vs[:1])):
        # print(v0, v1)
        if v0[1] > v1[1] and v0[0] > v1[0]:
            v1s = np.append(v1s, v1, axis=-1)
            if v0[1] > maxv0[1]:
                maxv0 = v0
    # print(v1s, maxv0)
                
    # interpolation: vs[:, 1] -> recall (x), vs[:, 0] -> time (y)
    vs = np.array(np.append(maxv0, v1s)).reshape(-1, 2) # 2-dim array
    f = interp1d(vs[:, 1], vs[:, 0])

    minx = np.min(vs[:, 1]) + 1e-6
    maxx = np.max(vs[:, 1]) - 1e-6
    x = np.arange(minx, maxx, 1.0) # the interval of interpolation: 1.0
    y = list(map(f, x))          # get time (y) by interpolation

    return x, y


# ------------------------------------------------------------------------------
def upper_bound_curve(xys, interval, is_sorted):
    '''
    get the time-ratio and precision-recall curves by convex hull and interpolation

    :params xys: 2-dim array (np.array)
    :params interval: the interval of interpolation (float)
    :params is_sorted: sort the convex hull or not (boolean)
    :returns: curve with interpolation
    '''
    # add noise and conduct convex hull to find the curve
    eps = np.random.normal(size=xys.shape) * 1e-6
    xys += eps
    # print(xys)
    
    xs = xys[:, 0]
    if len(xs) > 2 and xs[-1] > 0:
        hull = ConvexHull(xys)
        hull_vs = xys[hull.vertices]
        if is_sorted:
            hull_vs = np.array(sorted(hull_vs, key=lambda x:x[1]))
        print("hull_vs: ", hull_vs)

        # find max pair (maxv0) and min pairs (v1s) from the convex hull
        v1s = []
        maxv0 = [-1, -1]
        for v0, v1 in zip(hull_vs, chain(hull_vs[1:], hull_vs[:1])):
            # print(v0, v1)
            if v0[1] > v1[1] and v0[0] < v1[0]:
                v1s = np.append(v1s, v1, axis=-1)
                if v0[1] > maxv0[1]:
                    maxv0 = v0
        print(v1s, maxv0)

        # interpolation: vs[:, 1] -> recall (x), vs[:, 0] -> time (y)
        vs = np.array(np.append(maxv0, v1s)).reshape(-1, 2) # 2-dim array
        if len(vs) >= 2:
            f = interp1d(vs[:, 1], vs[:, 0])

            minx = np.min(vs[:, 1]) + 1e-6
            maxx = np.max(vs[:, 1]) - 1e-6
            x = np.arange(minx, maxx, interval)
            y = list(map(f, x))          # get time (y) by interpolation

            return x, y
        else:
            return xys[:, 0], xys[:, 1]
    else:
        return xys[:, 0], xys[:, 1]


# ------------------------------------------------------------------------------
def lower_bound_curve2(xys):
    '''
    get the querytime-indexsize and querytime-indextime curve by convex hull

    :params xys: 2-dim array (np.array)
    :returns: querytime-indexsize and querytime-indextime curve
    '''
    # add noise and conduct convex hull to find the curve
    eps = np.random.normal(size=xys.shape) * 1e-6
    xys += eps
    # print(xys)

    xs = xys[:, 0]
    if len(xs) > 2 and xs[-1] > 0:
        # conduct convex hull to find the curve
        hull = ConvexHull(xys)
        hull_vs = xys[hull.vertices]
        # print("hull_vs: ", hull_vs)
        
        ret_vs = []
        for v0, v1, v2 in zip(chain(hull_vs[-1:], hull_vs[:-1]), hull_vs, \
            chain(hull_vs[1:], hull_vs[:1])):

            # print(v0, v1, v2)
            if v0[0] < v1[0] or v1[0] < v2[0]:
                ret_vs = np.append(ret_vs, v1, axis=-1)

        # sort the results in ascending order of x without interpolation
        ret_vs = ret_vs.reshape((-1, 2))
        ret_vs = np.array(sorted(ret_vs, key=lambda x:x[0]))

        return ret_vs[:, 0], ret_vs[:, 1]
    else:
        return xys[:, 0], xys[:, 1]


# ------------------------------------------------------------------------------
def plot_time_recall(chosen_top_k, methods, input_folder, output_folder):
    '''
    draw the querytime-recall curve for all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    fig_width, fig_height = calc_width_and_height(len(datasets), 1)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure
    
    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up each sub-figure
        ax = plt.subplot(1, len(datasets), di+1)
        plt.title(dataset_label)            # title
        plt.xlim(0, 100)                    # limit (or range) of x-axis
        plt.xlabel('Recall (%)')            # label of x-axis
        if di == 0:                         # add label of y-axis at 1st dataset
            plt.ylabel('Query Time (ms)')

        miny = 1e9
        maxy = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):
            
            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get time-recall results
            time_recalls = []
            for _,res in parse_res(filename, chosen_top_k):
                time_recalls += [[gettime(res), getrecall(res)]]

            time_recalls = np.array(time_recalls)
            # print(time_recalls)

            # get the time-recall curve by convex hull and interpolation, where 
            # lower_recalls -> x, lower_times -> y
            lower_recalls, lower_times = lower_bound_curve(time_recalls) 
            miny = min(miny, np.min(lower_times))
            maxy = max(maxy, np.max(lower_times)) 
            ax.semilogy(lower_recalls, lower_times, '-', color=method_color, 
                marker=method_marker, label=method_label if di==0 else "", 
                markevery=10, markerfacecolor='none', markersize=7, 
                zorder=len(methods)-method_idx)

        # set up the limit (or range) of y-axis
        plt_helper.set_y_axis_log10(ax, miny, maxy)
    
    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'time_recall')
    plt.show()


# ------------------------------------------------------------------------------
def plot_fraction_recall(chosen_top_k, methods, input_folder, output_folder):
    '''
    draw the fraction-recall curve for all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    fig_width, fig_height = calc_width_and_height(len(datasets), 1)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure
    
    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up each sub-figure
        ax = plt.subplot(1, len(datasets), di+1)
        plt.title(dataset_label)            # title
        plt.xlim(0, 100)                    # limit (or range) of x-axis
        plt.xlabel('Recall (%)')            # label of x-axis
        if di == 0:                         # add label of y-axis at 1st dataset
            plt.ylabel('Fraction (%)')

        miny = 1e9
        maxy = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):
            
            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get fraction-recall results
            fraction_recalls = []
            for _,res in parse_res(filename, chosen_top_k):
                fraction_recalls += [[getfraction(res), getrecall(res)]]

            fraction_recalls = np.array(fraction_recalls)
            # print(fraction_recalls)

            # get the fraction-recall curve by convex hull and interpolation, where 
            # lower_recalls -> x, lower_times -> y
            # print('fraction_recall!!!!\n', fraction_recalls)
            lower_recalls, lower_fractions = lower_bound_curve(fraction_recalls) 
            miny = min(miny, np.min(lower_fractions))
            maxy = max(maxy, np.max(lower_fractions)) 
            ax.semilogy(lower_recalls, lower_fractions, '-', color=method_color, 
                marker=method_marker, label=method_label if di==0 else "", 
                markevery=10, markerfacecolor='none', markersize=7, 
                zorder=len(methods)-method_idx)

        # set up the limit (or range) of y-axis
        plt_helper.set_y_axis_log10(ax, miny, maxy)
    
    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'fraction_recall')
    plt.show()
    

# ------------------------------------------------------------------------------
def plot_precision_recall(chosen_top_k, methods, input_folder, output_folder):
    '''
    draw the precision-recall curve for all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params methods: a list of method (list)
    :returns: None
    '''
    fig_width, fig_height = calc_width_and_height(len(datasets), 1)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure
    
    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up each sub-figure
        ax = plt.subplot(1, len(datasets), di+1)
        plt.title(dataset_label)            # title
        plt.xlim(0, 100)                    # limit (or range) of x-axis
        plt.xlabel('Recall (%)')            # label of x-axis
        if di == 0:                         # add label of y-axis for the 1st dataset
            plt.ylabel('Precision (%)')

        miny = 1e9
        maxy = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):
            
            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get precision-recall results
            precision_recalls = []
            for _,res in parse_res(filename, chosen_top_k):
                precision = getprecision(res)
                recall    = getrecall(res)
                if (recall > 0 and precision > 0): 
                    precision_recalls += [[precision, recall]]

            precision_recalls = np.array(precision_recalls)
            # print(precision_recalls)

            # get the time-recall curve by convex hull and interpolation, where 
            upper_recalls, upper_precisions = upper_bound_curve(precision_recalls, 1.0, True) 
            if len(upper_recalls) > 0:
                miny = min(miny, np.min(upper_precisions))
                maxy = max(maxy, np.max(upper_precisions)) 
                ax.semilogy(upper_recalls, upper_precisions, '-', 
                    color=method_color, marker=method_marker, 
                    label=method_label if di==0 else "", markevery=10, 
                    markerfacecolor='none', markersize=7, 
                    zorder=len(methods)-method_idx)

        # set up the limit (or range) of y-axis
        plt_helper.set_y_axis_log10(ax, miny, maxy)
    
    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'precision_recall')
    plt.show()


# ------------------------------------------------------------------------------
def plot_time_recall_ratio(chosen_top_k, methods, input_folder, output_folder):
    '''
    draw the querytime-recall curves and querytime-ratio curves for all methods 
    on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    n_datasets = len(datasets)
    fig_width, fig_height = calc_width_and_height(n_datasets, 2)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure
    
    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up two sub-figures
        ax_recall = plt.subplot(2, n_datasets, di+1)
        plt.title(dataset_label)            # title
        plt.xlabel('Recall (%)')            # label of x-axis
        plt.xlim(0, 100)
        
        ax_ratio = plt.subplot(2, n_datasets, n_datasets+di+1)
        plt.xlabel('Ratio')
        plt.xlim(1.0, 11.0) 
        plt.xticks([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])

        if di == 0:
            ax_recall.set_ylabel('Query Time (ms)')
            ax_ratio.set_ylabel('Query Time (ms)')

        miny = 1e9
        maxy = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):

            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get querytime-recall and querytime-ratio results from disk
            time_recalls = []
            time_ratios = []
            for _,res in parse_res(filename, chosen_top_k):
                time_recalls += [[gettime(res), getrecall(res)]]
                time_ratios  += [[gettime(res), getratio(res)]]

            time_recalls = np.array(time_recalls)
            time_ratios  = np.array(time_ratios)
            # print(time_recalls, time_ratios)
            
            # get the querytime-recall curve by convex hull and interpolation
            lower_recalls, lower_times = lower_bound_curve(time_recalls)
            ax_recall.semilogy(lower_recalls, lower_times, '-', 
                color=method_color, marker=method_marker, 
                label=method_label if di==0 else "", markevery=10, 
                markerfacecolor='none', markersize=10)
            
            miny = min(miny, np.min(lower_times))
            maxy = max(maxy, np.max(lower_times))

            # get the querytime-ratio curve by convex hull
            upper_ratios, upper_times = upper_bound_curve(time_ratios, 0.2, False)
            ax_ratio.semilogy(upper_ratios, upper_times, '-', 
                color=method_color, marker=method_marker, label="", 
                markevery=5, markerfacecolor='none', markersize=10, 
                zorder=len(methods)-method_idx)
            
            miny = min(miny, np.min(upper_times))
            maxy = max(maxy, np.max(upper_times))
        
        # set up the limit (or range) of y-axis
        plt_helper.set_y_axis_log10(ax_recall, miny, maxy)
        plt_helper.set_y_axis_log10(ax_ratio,  miny, maxy)
    
    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'time_recall_ratio')
    

# ------------------------------------------------------------------------------
def plot_time_index(chosen_top_k, recall_level, methods, input_folder, output_folder):
    '''
    draw the querytime-indexsize curves and querytime-indexingtime curves for 
    all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params recall_level: recall value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    n_datasets = len(datasets)
    fig_width, fig_height = calc_width_and_height(n_datasets, 2)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure

    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up two sub-figures
        ax_size = plt.subplot(2, n_datasets, di+1)
        plt.title(dataset_label)            # title
        plt.xlabel('Index Size (MB)')       # label of x-axis

        ax_time = plt.subplot(2, n_datasets, n_datasets+di+1)
        plt.xlabel('Indexing Time (Seconds)')   # label of x-axis

        if di == 0:
            ax_size.set_ylabel('Query Time (ms)')
            ax_time.set_ylabel('Query Time (ms)')

        miny = 1e9
        maxy = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):

            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get all results from disk
            chosen_ks_dict = defaultdict(list)
            for _,res in parse_res(filename, chosen_top_k):
                query_time = gettime(res)
                recall     = getrecall(res)
                chosen_k = getindexingtime(res)
                index_size = getindexsize(res)
                
                chosen_ks_dict[(chosen_k, index_size)] += [[recall, query_time]]

            # get querytime-indexsize and querytime-indexingtime results if its 
            # recall is higher than recall_level 
            chosen_ks, index_sizes, querytimes_at_recall = [], [], []
            for (chosen_k, index_size), recall_querytimes_ in chosen_ks_dict.items():
                # add [[0, 0]] for interpolation
                recall_querytimes_ = np.array([[0, 0]] +recall_querytimes_)

                recalls, query_times = lower_bound_curve2(recall_querytimes_)
                # recalls = recall_querytimes[:, 0]
                # query_times = recall_querytimes[:, 1]

                # print('recall_querytimes', recall_querytimes)
                
                if np.max(recalls) > recall_level:                    
                    # get the estimated time at recall level by interpolation 
                    f = interp1d(recalls, query_times)
                    querytime_at_recall = f(recall_level)
                    # print('iit', chosen_k, index_size, querytime_at_recall)

                    # update results
                    chosen_ks += [chosen_k]
                    index_sizes += [index_size]
                    querytimes_at_recall += [querytime_at_recall]

                    # print('recall_querytimes!!!!\n', recall_querytimes)
                    print('interp, ', querytime_at_recall, index_size, chosen_k)
            
            chosen_ks = np.array(chosen_ks)
            index_sizes = np.array(index_sizes)
            querytimes_at_recall = np.array(querytimes_at_recall)
          
            # get the querytime-indexsize curve by convex hull
            indextimes_qtimes = np.zeros(shape=(len(index_sizes), 2))
            indextimes_qtimes[:, 0] = index_sizes
            indextimes_qtimes[:, 1] = querytimes_at_recall

            # print('qtimes', indextimes_qtimes)

            lower_isizes, lower_qtimes = lower_bound_curve2(indextimes_qtimes)
            if len(lower_isizes) > 0:
                # print(method, lower_isizes, lower_qtimes)
                ax_size.semilogy(lower_isizes, lower_qtimes, '-', color=method_color, 
                    marker=method_marker, label=method_label if di==0 else "", 
                    markerfacecolor='none', markersize=10)

                miny = min(miny, np.min(lower_qtimes))
                maxy = max(maxy, np.max(lower_qtimes)) 
                
                # get the querytime-indextime curve by convex hull
                itime_qtimes = np.zeros(shape=(len(chosen_ks), 2))
                itime_qtimes[:, 0] = chosen_ks
                itime_qtimes[:, 1] = querytimes_at_recall

                lower_itimes, lower_qtimes = lower_bound_curve2(itime_qtimes)
                # print(method, lower_itimes, lower_qtimes)
                ax_time.semilogy(lower_itimes, lower_qtimes, '-', color=method_color, 
                    marker=method_marker, label="", markerfacecolor='none', 
                    markersize=10, zorder=len(methods)-method_idx)

                miny = min(miny, np.min(lower_qtimes))
                maxy = max(maxy, np.max(lower_qtimes))
                
        # set up the limit (or range) of y-axis 
        plt_helper.set_y_axis_log10(ax_time, miny, maxy)
        plt_helper.set_y_axis_log10(ax_size, miny, maxy)

    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'time_index')


# ------------------------------------------------------------------------------
def plot_time_index_time(chosen_top_k, recall_level, methods, input_folder, 
    output_folder):
    '''
    draw the querytime-indexsize curves and querytime-indexingtime curves for 
    all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params recall_level: recall value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    n_datasets = len(datasets)
    fig_width, fig_height = calc_width_and_height(n_datasets, 1)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure

    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up sub-figure
        ax_time = plt.subplot(1, n_datasets, di+1)
        plt.title(dataset_label)            # title
        plt.xlabel('Indexing Time (Seconds)')   # label of x-axis

        if di == 0:
            ax_time.set_ylabel('Query Time (ms)')

        miny = 1e9
        maxy = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):

            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get all results from disk
            chosen_ks_dict = defaultdict(list)
            for _,res in parse_res(filename, chosen_top_k):
                query_time = gettime(res)
                recall     = getrecall(res)
                chosen_k = getindexingtime(res)
                index_size = getindexsize(res)
                
                chosen_ks_dict[(chosen_k, index_size)] += [[recall, query_time]]

            # get querytime-indexsize and querytime-indexingtime results if its 
            # recall is higher than recall_level 
            chosen_ks, index_sizes, querytimes_at_recall = [], [], []
            for (chosen_k, index_size), recall_querytimes_ in chosen_ks_dict.items():
                # add [[0, 0]] for interpolation
                recall_querytimes_ = np.array([[0, 0]] +recall_querytimes_)
                recalls, query_times = lower_bound_curve2(recall_querytimes_)
                
                if np.max(recalls) > recall_level:                    
                    # get the estimated time at recall level by interpolation 
                    f = interp1d(recalls, query_times)
                    querytime_at_recall = f(recall_level)
                    # print('iit', chosen_k, index_size, querytime_at_recall)

                    # update results
                    chosen_ks += [chosen_k]
                    index_sizes += [index_size]
                    querytimes_at_recall += [querytime_at_recall]

                    # print('recall_querytimes!!!!\n', recall_querytimes)
                    print('interp, ', querytime_at_recall, index_size, chosen_k)
            
            chosen_ks = np.array(chosen_ks)
            index_sizes = np.array(index_sizes)
            querytimes_at_recall = np.array(querytimes_at_recall)
          
            # get the querytime-indexsize curve by convex hull
            indextimes_qtimes = np.zeros(shape=(len(index_sizes), 2))
            indextimes_qtimes[:, 0] = index_sizes
            indextimes_qtimes[:, 1] = querytimes_at_recall
            # print('qtimes', indextimes_qtimes)

            lower_isizes, lower_qtimes = lower_bound_curve2(indextimes_qtimes)
            if len(lower_isizes) > 0:                
                # get the querytime-indextime curve by convex hull
                itime_qtimes = np.zeros(shape=(len(chosen_ks), 2))
                itime_qtimes[:, 0] = chosen_ks
                itime_qtimes[:, 1] = querytimes_at_recall

                lower_itimes, lower_qtimes = lower_bound_curve2(itime_qtimes)
                # print(method, lower_itimes, lower_qtimes)
                ax_time.semilogy(lower_itimes, lower_qtimes, '-', 
                    color=method_color, marker=method_marker, 
                    label=method_label if di==0 else "", 
                    markerfacecolor='none', markersize=10, 
                    zorder=len(methods)-method_idx)

                miny = min(miny, np.min(lower_qtimes))
                maxy = max(maxy, np.max(lower_qtimes))
                
        # set up the limit (or range) of y-axis 
        plt_helper.set_y_axis_log10(ax_time, miny, maxy)

    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'time_index_time')


# ------------------------------------------------------------------------------
def plot_time_k(chosen_top_ks, recall_level, methods, input_folder, 
    output_folder):
    '''
    draw the querytime-indexsize curves and querytime-indexingtime curves for 
    all methods on all datasets 

    :params chosen_top_ks: top_k value for drawing figure (list)
    :params recall_level: recall value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    n_datasets = len(datasets)
    fig_width, fig_height = calc_width_and_height(n_datasets, 1)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure

    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up sub-figure
        ax_time = plt.subplot(1, n_datasets, di+1)
        plt.title(dataset_label)            # title
        plt.xlabel('$k$')   # label of x-axis

        if di == 0:
            ax_time.set_ylabel('Query Time (ms)')

        miny = 1e9
        maxy = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):

            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get all results from disk
            chosen_ks_dict = defaultdict(list)
            for chosen_top_k in chosen_top_ks:
                for _,res in parse_res(filename, chosen_top_k):
                    query_time = gettime(res)
                    recall     = getrecall(res)
                    
                    chosen_ks_dict[chosen_top_k] += [[recall, query_time]]

            # get querytime-indexsize and querytime-indexingtime results if its 
            # recall is higher than recall_level 
            chosen_ks, querytimes_at_recall = [], []
            for chosen_k, recall_querytimes_ in chosen_ks_dict.items():
                # add [[0, 0]] for interpolation
                recall_querytimes_ = np.array([[0, 0]] + recall_querytimes_)
                recalls, query_times = lower_bound_curve2(recall_querytimes_)
                
                if np.max(recalls) > recall_level:                    
                    # get the estimated time at recall level by interpolation 
                    f = interp1d(recalls, query_times)
                    querytime_at_recall = f(recall_level)
                    # print('iit', chosen_k, index_size, querytime_at_recall)

                    # update results
                    chosen_ks += [chosen_k]
                    querytimes_at_recall += [querytime_at_recall]

            chosen_ks = np.array(chosen_ks)
            querytimes_at_recall = np.array(querytimes_at_recall)

            ax_time.semilogy(chosen_ks, querytimes_at_recall, '-', 
                    color=method_color, marker=method_marker, 
                    label=method_label if di==0 else "", 
                    markerfacecolor='none', markersize=10, 
                    zorder=len(methods)-method_idx)

            # ax_time.set_xticklabels(['%d'% k for k in chosen_ks], minor=False, rotation=0)

            miny = min(miny, np.min(querytimes_at_recall))
            maxy = max(maxy, np.max(querytimes_at_recall))
                
        # set up the limit (or range) of y-axis 
        plt_helper.set_y_axis_log10(ax_time, miny, maxy)

    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'time_k')


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    chosen_top_k   = 10
    recall_level   = 50
    input_folders  = ["results/"]

    chosen_top_ks  = [1,5,10,20,50,100]
    
    output_folders = ["figures/competitors/"]
    for input_folder, output_folder in zip(input_folders, output_folders):
        methods = ['MFH_Sampling', 'FH_Sampling', 'NH_LCCS_Sampling', 'BH', 'MH', 
            'Random_Scan', 'Sorted_Scan']
        plot_time_recall(chosen_top_k, methods, input_folder, output_folder)
        plot_fraction_recall(chosen_top_k, methods, input_folder, output_folder)
        plot_time_index(chosen_top_k, recall_level, methods, input_folder, output_folder)
        plot_time_k(chosen_top_ks, recall_level, methods, input_folder, output_folder)
    
    output_folders = ["figures/sampling/"]
    for input_folder, output_folder in zip(input_folders, output_folders):
        methods = ['MFH_Sampling', 'FH_Sampling', 'NH_LCCS_Sampling', 'MFH', 'FH', 'NH_LCCS']
        plot_time_recall(chosen_top_k, methods, input_folder, output_folder)
        plot_fraction_recall(chosen_top_k, methods, input_folder, output_folder)
        plot_time_index(chosen_top_k, recall_level, methods, input_folder, output_folder)
        plot_time_index_time(chosen_top_k, recall_level, methods, input_folder, output_folder)

