import os
import re
import numpy as np
import matplotlib.pylab as plt

from scipy.spatial     import ConvexHull
from itertools         import chain
from scipy.interpolate import interp1d
from collections       import defaultdict

from plot      import get_filename, parse_res
from plot      import get_cand, get_m, get_l, get_s, get_time, get_recall
from plot      import method_colors, method_markers, dataset_labels_map
from plot_util import PlotHelper


# ------------------------------------------------------------------------------
def plot_nh_t(chosen_top_k, datasets, input_folder, output_folder, fig_width=6.5, fig_height=6.0):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=1.2, hspace=0.37)
    
    method = 'NH_LCCS_Sampling'
    
    for di, dataset in enumerate(datasets):
        ax = plt.subplot(1, len(datasets), di+1)
        ax.set_xlabel(r'Recall (%)')
        if di == 0:
            ax.set_ylabel(r'Query Time (ms)')
        ax.set_title('%s' % dataset_labels_map[dataset])
        
        # get file name for this method on this dataset
        filename = get_filename(input_folder, dataset, method)
        print(filename, method, dataset)
        
        fix_s=2
        data = []
        for record in parse_res(filename, chosen_top_k):
            # print(record)
            m      = get_m(record)
            s      = get_s(record)
            cand   = get_cand(record)
            time   = get_time(record)
            recall = get_recall(record)

            if s == fix_s:
                print(m, s, cand, time, recall)
                data += [[m, s, cand, time, recall]]
        data = np.array(data)

        
        ms = [8, 16, 32, 64, 128, 256]
        maxy = -1e9
        miny = 1e9
        for color, marker, m in zip(method_colors, method_markers, ms):
            data_mp = data[data[:, 0]==m]
            # print(m, data_mp)
            
            plt.semilogy(data_mp[:, -1], data_mp[:, -2], marker=marker, 
                label='$t=%d$'%(m) if di==0 else "", c=color, 
                markerfacecolor='none', markersize=7)
            miny = min(miny, np.min(data_mp[:,-2]) )
            maxy = max(maxy, np.max(data_mp[:,-2]) ) 
        plt.xlim(0, 100)
        # print(dataset, distance, miny, maxy)
        plt_helper.set_y_axis_log10(ax, miny, maxy)
    
    plt_helper.plot_fig_legend(ncol=3)
    plt_helper.plot_and_save(output_folder, 'varying_nh_t')


# ------------------------------------------------------------------------------
def plot_fh_m(chosen_top_k, datasets, input_folder, output_folder, fig_width=6.5, fig_height=6.0):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=1.2, hspace=0.37)
    
    method = 'MFH_Sampling'
    
    for di, dataset in enumerate(datasets):
        ax = plt.subplot(1, len(datasets), di+1)
        ax.set_xlabel(r'Recall (%)')
        if di == 0:
            ax.set_ylabel(r'Query Time (ms)')
        ax.set_title('%s' % dataset_labels_map[dataset])
        
        # get file name for this method on this dataset
        filename = get_filename(input_folder, dataset, method)
        print(filename, method, dataset)
        
        fix_l=4
        fix_s=2
        data = []
        for record in parse_res(filename, chosen_top_k):
            # print(record)
            m      = get_m(record)
            l      = get_l(record)
            s      = get_s(record)
            cand   = get_cand(record)
            time   = get_time(record)
            recall = get_recall(record)

            if l == fix_l and s == fix_s:
                print(m, l, s, cand, time, recall)
                data += [[m, l, s, cand, time, recall]]
        data = np.array(data)

        
        ms = [8, 16, 32, 64, 128, 256]
        maxy = -1e9
        miny = 1e9
        for color, marker, m in zip(method_colors, method_markers, ms):
            data_mp = data[data[:, 0]==m]
            # print(m, data_mp)
            
            plt.semilogy(data_mp[:, -1], data_mp[:, -2], marker=marker, 
                label='$m=%d$'%(m) if di==0 else "", c=color, 
                markerfacecolor='none', markersize=7)
            miny = min(miny, np.min(data_mp[:,-2]) )
            maxy = max(maxy, np.max(data_mp[:,-2]) ) 
        plt.xlim(0, 100)
        # print(dataset, distance, miny, maxy)
        plt_helper.set_y_axis_log10(ax, miny, maxy)
    
    plt_helper.plot_fig_legend(ncol=3)
    plt_helper.plot_and_save(output_folder, 'varying_fh_m')


# ------------------------------------------------------------------------------
def plot_fh_l(chosen_top_k, datasets, input_folder, output_folder, fig_width=6.5, fig_height=6.0):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.8, hspace=0.37)
    
    method = 'MFH_Sampling'
    
    for di, dataset in enumerate(datasets):
        ax = plt.subplot(1, len(datasets), di+1)
        ax.set_xlabel(r'Recall (%)')
        if di == 0:
            ax.set_ylabel(r'Query Time (ms)')
        ax.set_title('%s' % dataset_labels_map[dataset])
        
        # get file name for this method on this dataset
        filename = get_filename(input_folder, dataset, method)
        print(filename, method, dataset)
        
        fix_m=16
        fix_s=2
        data = []
        for record in parse_res(filename, chosen_top_k):
            # print(record)
            m      = get_m(record)
            l      = get_l(record)
            s      = get_s(record)
            cand   = get_cand(record)
            time   = get_time(record)
            recall = get_recall(record)

            if m == fix_m and s == fix_s:
                print(m, l, s, cand, time, recall)
                data += [[m, l, s, cand, time, recall]]
        data = np.array(data)

        
        ls = [2, 4, 6, 8, 10]
        maxy = -1e9
        miny = 1e9
        for color, marker, l in zip(method_colors, method_markers, ls):
            data_mp = data[data[:, 1]==l]
            # print(m, data_mp)
            
            plt.semilogy(data_mp[:, -1], data_mp[:, -2], marker=marker, 
                label='$l=%d$'%(l) if di==0 else "", c=color, 
                markerfacecolor='none', markersize=7)
            miny = min(miny, np.min(data_mp[:,-2]) )
            maxy = max(maxy, np.max(data_mp[:,-2]) ) 
        plt.xlim(0, 100)
        # print(dataset, distance, miny, maxy)
        plt_helper.set_y_axis_log10(ax, miny, maxy)
    
    plt_helper.plot_fig_legend(ncol=5)
    plt_helper.plot_and_save(output_folder, 'varying_fh_l')


# ------------------------------------------------------------------------------
def plot_fh_s(chosen_top_k, datasets, input_folder, output_folder, fig_width=6.5, fig_height=6.0):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.8, hspace=0.37)
    
    method = 'MFH_Sampling'
    
    for di, dataset in enumerate(datasets):
        ax = plt.subplot(1, len(datasets), di+1)
        ax.set_xlabel(r'Recall (%)')
        if di == 0:
            ax.set_ylabel(r'Query Time (ms)')
        ax.set_title('%s' % dataset_labels_map[dataset])
        
        # get file name for this method on this dataset
        filename = get_filename(input_folder, dataset, method)
        print(filename, method, dataset)
        
        fix_m=16
        fix_l=4
        data = []
        for record in parse_res(filename, chosen_top_k):
            # print(record)
            m      = get_m(record)
            l      = get_l(record)
            s      = get_s(record)
            cand   = get_cand(record)
            time   = get_time(record)
            recall = get_recall(record)

            if m == fix_m and l == fix_l:
                print(m, l, s, cand, time, recall)
                data += [[m, l, s, cand, time, recall]]
        data = np.array(data)
        
        ss = [1, 2, 4, 8]
        maxy = -1e9
        miny = 1e9
        for color, marker, s in zip(method_colors, method_markers, ss):
            data_mp = data[data[:, 2]==s]
            # print(m, data_mp)
            
            plt.semilogy(data_mp[:, -1], data_mp[:, -2], marker=marker, 
                label='$\lambda=%d d$'%(s) if di==0 else "", c=color, 
                markerfacecolor='none', markersize=7)
            miny = min(miny, np.min(data_mp[:,-2]) )
            maxy = max(maxy, np.max(data_mp[:,-2]) ) 
        plt.xlim(0, 100)
        # print(dataset, distance, miny, maxy)
        plt_helper.set_y_axis_log10(ax, miny, maxy)
    
    plt_helper.plot_fig_legend(ncol=4)
    plt_helper.plot_and_save(output_folder, 'varying_fh_s')

# ------------------------------------------------------------------------------
def plot_nh_s(chosen_top_k, datasets, input_folder, output_folder, fig_width=6.5, fig_height=6.0):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.8, hspace=0.37)
    
    method = 'NH_LCCS_Sampling'
    
    for di, dataset in enumerate(datasets):
        ax = plt.subplot(1, len(datasets), di+1)
        ax.set_xlabel(r'Recall (%)')
        if di == 0:
            ax.set_ylabel(r'Query Time (ms)')
        ax.set_title('%s' % dataset_labels_map[dataset])
        
        # get file name for this method on this dataset
        filename = get_filename(input_folder, dataset, method)
        print(filename, method, dataset)
        
        fix_m=256
        data = []
        for record in parse_res(filename, chosen_top_k):
            # print(record)
            m      = get_m(record)
            s      = get_s(record)
            cand   = get_cand(record)
            time   = get_time(record)
            recall = get_recall(record)

            if m == fix_m:
                print(m, s, cand, time, recall)
                data += [[m, s, cand, time, recall]]
        data = np.array(data)
        
        ss = [1, 2, 4, 8]
        maxy = -1e9
        miny = 1e9
        for color, marker, s in zip(method_colors, method_markers, ss):
            data_mp = data[data[:, 1]==s]
            # print(m, data_mp)
            
            plt.semilogy(data_mp[:, -1], data_mp[:, -2], marker=marker, 
                label='$\lambda=%d d$'%(s) if di==0 else "", c=color, 
                markerfacecolor='none', markersize=7)
            miny = min(miny, np.min(data_mp[:,-2]) )
            maxy = max(maxy, np.max(data_mp[:,-2]) ) 
        plt.xlim(0, 100)
        # print(dataset, distance, miny, maxy)
        plt_helper.set_y_axis_log10(ax, miny, maxy)
    
    plt_helper.plot_fig_legend(ncol=4)
    plt_helper.plot_and_save(output_folder, 'varying_nh_s')

# ------------------------------------------------------------------------------
if __name__ == '__main__':

    chosen_top_k = 10
    input_folder = "results/"
    output_folder = "figures/param/"
    datasets = ['Yelp', 'GloVe100']

    plot_nh_t(chosen_top_k, datasets, input_folder, output_folder, fig_height=3.4)
    plot_nh_s(chosen_top_k, datasets, input_folder, output_folder, fig_height=3.0)
    plot_fh_m(chosen_top_k, datasets, input_folder, output_folder, fig_height=3.4)
    plot_fh_l(chosen_top_k, datasets, input_folder, output_folder, fig_height=3.0)
    plot_fh_s(chosen_top_k, datasets, input_folder, output_folder, fig_height=3.0)
