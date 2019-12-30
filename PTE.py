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

def plot_all_PTE_cases(cases, w=1050, h=350):
	silence(FIXED_SIZING_MODE, True)
	plot_dict = {}
	TOOLTIPS = [("(x,y)", "($x{1.1}, $y{int})")]
	for case in cases.values():
		for ch in case.df.columns:
			if 'channel_5' in ch:
				# print(ch)
				plot_dict = clo.plot_size_standard(case, ch, plot_dict, w, h, ss_channel_num=5)
			else:
				ch_num = re.findall(r'channel_\d$', ch)[0]
				p = figure(tools='pan,wheel_zoom,reset',title=ch, width=w, height=h, x_axis_label='fragment size', y_axis_label='RFU', tooltips=TOOLTIPS, x_range=(1000,5000))
				x = case.df.index.to_list()
				y = case.df[ch].to_list()
				p.line(x, y, line_width=0.5, color=clo.channel_colors.get(ch_num, 'blue'))
				plot_dict[ch] = p
	plots = column([plot_dict[ch] for ch in sorted(plot_dict)])
	show(plots)

def plot_PTE_case(case, plot_dict, w=1050, h=200, ss_channel_num=5):
	silence(FIXED_SIZING_MODE, True)
	TOOLTIPS = [("(x,y)", "($x{1.1}, $y{int})")]
	ss_channel = 'channel_' + str(ss_channel_num)
	ch_list = [ch for ch in case.df.columns if 'x_fitted' not in ch]
	for ch in ch_list:
		ch_num = re.findall(r'channel_\d$', ch)[0]
		x_col_name = 'x_fitted_' + re.sub(r'channel_\d',ss_channel, ch)
		p = figure(tools='pan,wheel_zoom,reset',title=ch, width=w, height=h, x_axis_label='fragment size', y_axis_label='RFU', tooltips=TOOLTIPS)
		if ss_channel not in ch:
			p.x_range = Range1d(75,400)
			x = case.df[x_col_name].to_list()
			y = case.df[ch].to_list()
			p.line(x, y, line_width=0.5, color=clo.channel_colors.get(ch_num, 'blue'))
			plot_dict[ch] = p
		else:
			plot_dict = clo.plot_size_standard(case, ch, plot_dict, w, h=400, ss_channel_num=5)
	# print('len(plot_dict.values()) = {}'.format(len(plot_dict.values())))
	# print(type(plot_dict.values()))
	return plot_dict

def main():
	owd = os.getcwd()	# original working directory
	path = os.path.abspath(sys.argv[1])
	os.chdir(path)
	cases = organize_PTE_files(path)
	plot_dict = {}
	for name, case in cases.items():
		print('working on {}'.format(case.name))
		case = gather_PTE_case_data(case, path)
		# print(case.df)
		# for ch in case.df.columns:
		# 	case = clo.baseline_correction(case, ch, distance=1)
		case = clo.baseline_correction(case, ss_channel_num=5, distance=5)
		case = clo.size_standard(case, channel='channel_5')
		case = clo.local_southern(case)
		plot_dict = plot_PTE_case(case, plot_dict, w=1050, h=200)

	# sort the plots. SCL first, channel + repeat after, followed by their size standards.
	plot_keys = sorted([key for key in plot_dict.keys() if 'SCL' not in key])
	scl_keys = sorted([key for key in plot_dict.keys() if 'SCL' in key])
	plot_keys = [*scl_keys, *plot_keys]
	plots = column([plot_dict[ch] for ch in plot_keys], sizing_mode='fixed')

	case_html = case.name + '.html'
	output_file(case_html)
	show(plots)
	save(plots)
	print('Saved {}'.format(case_html))


if __name__ == '__main__':
	main()


"""	Next steps
	1. Correctly label regions and peaks on the allelic ladder.
	2. Use allelic ladder to annotate sample peaks.
		a. May need to deal with stutter vs. true allelic peak
	3. Get area under each peak.
		a. Given a list of peaks, get their bases.
		b. Include stutter in area calculation or not? Won't matter so long as we are consistent. Easier to exclude stutter.
"""