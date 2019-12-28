#!/usr/bin/env python3

# Importing Packages
import os
import sys
import re
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.interpolate import InterpolatedUnivariateSpline
from itertools import combinations
from outliers import smirnov_grubbs as grubbs

from bokeh.io import output_file, show, save
from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.models import BoxAnnotation, Label, Range1d, WheelZoomTool, ResetTool, PanTool, LegendItem, Legend
from bokeh.core.validation.warnings import FIXED_SIZING_MODE
from bokeh.core.validation import silence

import clonality as clo


def gather_PTE_case_data(case, path):
	df = pd.read_csv(os.path.join(path, case.name+'.csv'))
	columns_to_drop = [c for c in df.columns if 'channel' not in c]
	df = df.drop(columns_to_drop, axis=1)
	df.name = case.name
	case.df = df
	return case

def organize_PTE_files(path):
	# construct case list
	csv_list = [f for f in os.listdir(path) if f.endswith('.csv')]
	case_names = [f[:-4] for f in csv_list]
	# make a dictionary of case names to case files
	cases = {case_name: clo.Case() for case_name in case_names}
	for case_name,c in cases.items():
		c.name = case_name
		c.ladder = {}
		c.rox500 = []
	return cases

def plot_PTE_case(case, w=1050, h=350):
	silence(FIXED_SIZING_MODE, True)
	plot_dict = {}
	TOOLTIPS = [("(x,y)", "($x{1.1}, $y{int})")]
	for ch in case.df.columns:
		ch_num = re.findall(r'channel_\d$', ch)[0]
		color = clo.channel_colors.get(ch_num, 'blue')
		p = figure(tools='pan,wheel_zoom,reset',title=ch, width=w, height=h, x_axis_label='fragment size', y_axis_label='RFU', x_range=(0,5000), tooltips=TOOLTIPS)
		x = case.df.index.to_list()
		y = case.df[ch].to_list()
		p.line(x, y, line_width=0.5, color=color)
		plot_dict[ch] = p
	# print('len(plot_dict.values()) = {}'.format(len(plot_dict.values())))
	# print(type(plot_dict.values()))
	plots = column([p for p in plot_dict.values()])
	show(plots)

def plot_all_PTE_cases(cases, w=1050, h=350):
	silence(FIXED_SIZING_MODE, True)
	plot_dict = {}
	TOOLTIPS = [("(x,y)", "($x{1.1}, $y{int})")]
	for case in cases.values():
		for ch in case.df.columns:
			ch_num = re.findall(r'channel_\d$', ch)[0]
			color = clo.channel_colors.get(ch_num, 'blue')
			p = figure(tools='pan,wheel_zoom,reset',title=ch, width=w, height=h, x_axis_label='fragment size', y_axis_label='RFU', tooltips=TOOLTIPS, x_range=(1000,5000))
			
			x = case.df.index.to_list()
			y = case.df[ch].to_list()
			p.line(x, y, line_width=0.5, color=color)
			plot_dict[ch] = p
		# print('len(plot_dict.values()) = {}'.format(len(plot_dict.values())))
		# print(type(plot_dict.values()))
	plots = column([plot_dict[ch] for ch in sorted(plot_dict)])
	show(plots)


def main():
	owd = os.getcwd()	# original working directory
	path = os.path.abspath(sys.argv[1])
	os.chdir(path)
	cases = organize_PTE_files(path)
	for name, case in cases.items():
		print('working on {}'.format(case.name))
		case = gather_PTE_case_data(case, path)
		# print(case.df)
		for ch in case.df.columns:
			case = clo.baseline_correction(case, ch, distance=1)
		# plot_PTE_case(case, w=2000, h=500)
	plot_all_PTE_cases(cases)

	

if __name__ == '__main__':
	main()