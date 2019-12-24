#!/usr/bin/env python3

# Importing Packages
import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from time import strftime
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.interpolate import InterpolatedUnivariateSpline
from types import SimpleNamespace
from pprint import pprint
from itertools import combinations
import mpl_toolkits.mplot3d as m3d
import re
from outliers import smirnov_grubbs as grubbs

from bokeh.io import output_file, show, save
from bokeh.layouts import gridplot, column
from bokeh.plotting import figure
from bokeh.models import BoxAnnotation, Label, Range1d, WheelZoomTool, ResetTool, PanTool, WheelPanTool
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import FIXED_SIZING_MODE

timestr = strftime("%Y%m%d-%H%M%S")

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size

def autoscale_y(ax,margin=0.1):
	"""Courtesy of Stack Overflow: https://stackoverflow.com/questions/29461608/matplotlib-fixing-x-axis-scale-and-autoscale-y-axis

	This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
	ax -- a matplotlib axes object
	margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

	def get_bottom_top(line):
		xd = line.get_xdata()
		yd = line.get_ydata()
		lo,hi = ax.get_xlim()
		y_displayed = yd[((xd>lo) & (xd<hi))]
		print('y_displayed = {}'.format(y_displayed))
		h = np.max(y_displayed) - np.min(y_displayed)
		bot = np.min(y_displayed)-margin*h
		top = np.max(y_displayed)+margin*h
		return bot,top

	lines = ax.get_lines()
	bot,top = np.inf, -np.inf

	for line in lines:
		new_bot, new_top = get_bottom_top(line)
		if new_bot < bot: bot = new_bot
		if new_top > top: top = new_top

	ax.set_ylim(bot,top)

def autoscale_y_2(x_window_start, x_window_end, x_series, y_series):
	window_indices = [i for i, v in x_series.items() if v > x_window_start and v < x_window_end]
	y_max = y_series[window_indices].max()
	y_top = y_max + 0.1 * abs(y_max)
	if y_top < 200:
		y_top = 200
	y_min = y_series[window_indices].min()
	y_bot = y_min - 0.1 * abs(y_min)
	return y_bot, y_top

def pretty_name(c,t):
	if 'channel' in c:
		channel = re.findall(r'channel_\d$', c)[0]
		if 'repeat' in c:
			pc = '_'.join([t, channel, 'repeat'])
		else:
			pc = '_'.join([t, channel])
	else:
		pc = c
	return pc

def organize_files(path):
	tests = [
				'IGH-A', 'IGH-B', 'IGH-C', 'IGK-A', 'IGK-B',
				'TCRB-A', 'TCRB-B', 'TCRB-C', 'TCRG-A', 'TCRG-B',
				'SCL'
			]

	# construct case list
	csv_list = [f for f in os.listdir(path) if f.endswith('.csv')]
	# case_names_as_llt = [re.findall(r'(\d\dKD-\d\d\dM\d\d\d\d)(-R)*', x) for x in csv_list]	# 'llt' is 'list of lists of tuple'
	case_names_as_llt = [re.findall(r'(\d+KD-\d+M\d+)(-R)*', x) for x in csv_list]	# 'llt' is 'list of lists of tuple'
	case_names_as_ll = [list(lt[0]) for lt in case_names_as_llt if len(lt) > 0]	# ll is 'list of lists'
	case_names = {''.join(x) for x in case_names_as_ll}	# finally we have a set of unique strings

	# make a dictionary of case names to case files
	cd = {cn : { t : [f for f in csv_list if cn in f and t in f] for t in tests } for cn in case_names}
	cases = {cn: Case() for cn in case_names}
	for cn,c in cases.items():
		c.name = cn
		c.files = cd[cn]
		c.ladder = {}
		c.rox500 = []
		c.index_of_peaks_to_annotate = {}
		c.index_of_peaks_to_annotate = {}
		c.re_df = {}
	return cases

class Case(object):
	pass

def gather_case_data(case, case_name, path):
	df = pd.DataFrame()
	for t, files in case.files.items():
		for f in files:
			df_t = pd.read_csv(os.path.join(path,f))
			df_t.columns = [pretty_name(c,t) for c in df_t.columns]
			columns_to_drop = [c for c in df_t.columns if not (c.startswith('TCR') or c.startswith('IG') or c.startswith('SCL'))]
			df_t = df_t.drop(columns_to_drop, axis=1)
			df = pd.concat([df, df_t], axis=1, sort=False)
	df.name = case_name
	case.df = df
	return case

def local_southern(case, order=2):
	for ch_4, ladder in case.ladder.items():
		x_fitted = np.array([])
		for i in range(2,len(ladder)-1):
			x1 = ladder[i-2:i+1]
			y1 = case.rox500[i-2:i+1]
			polyx1 = np.poly1d(np.polyfit(x1,y1,deg=order))
			x2 = ladder[i-1:i+2]
			y2 = case.rox500[i-1:i+2]
			polyx2 = np.poly1d(np.polyfit(x2,y2,deg=order))
			if i == 2:
				x = range(case.df.index.tolist()[0], ladder[i])
			elif i == len(ladder)-2:
				x = range(ladder[i-1], case.df.index.tolist()[-1]+1)
				# print('x[0] = {}, x[-1] = {}'.format(x[0], x[-1]))
			else:
				x = range(ladder[i-1], ladder[i])
			y = np.average(np.array([polyx1(x), polyx2(x)]), axis=0)
			x_fitted = np.concatenate((x_fitted, y), axis=0)
		x_df = pd.DataFrame(x_fitted)
		# print('len(x_fitted) = {}'.format(len(x_fitted)))
		col_name = '_'.join(['x_fitted', ch_4])
		x_df.columns = [col_name]
		case.df = pd.concat([case.df, x_df], axis=1, sort=False)
	return case

# def plot_case_pdf(case):
# 	x_window_start = 75*10
# 	x_window_end = 400*10

# 	if use_timestamp:
# 		multipage = case.name + '_' + timestr + '.pdf'
# 	else:
# 		multipage = case.name + '.pdf'

# 	with PdfPages(multipage) as pdf:
# 		has_repeat = [ch for ch in case.re_df.keys() if '_'.join([ch, 'repeat']) in case.re_df.keys() and ch in channels_of_interest.keys()]
# 		# no_repeat = [ch for ch in case.re_df.keys() if ch in channels_of_interest.keys() and ch not in has_repeat]
# 		scl = [ch for ch in case.re_df.keys() if ch in channels_of_interest.keys() and 'SCL' in ch]

# 		for channel in has_repeat:
# 			channel_repeat = '_'.join([channel, 'repeat'])

# 			p, axs = plt.subplots(nrows=2, ncols=1)
# 			p.subplots_adjust(hspace=0.5)
# 			p.suptitle(case.name)

# 			for i, ch in enumerate([channel, channel_repeat]):
# 				for x_start,x_end in regions_of_interest[ch]:
# 					axs[i].axvspan(x_start*10, x_end*10, facecolor='black', alpha=0.05)
# 				df = case.re_df[ch]
# 				peaks_x = case.index_of_peaks_to_annotate[ch]
# 				axs[i].plot(df.iloc[peaks_x], 'o', color='black', fillstyle='none')
# 				# texts = [axs[i].text(x, 1.05 * df.iloc[x], str(x)) for x in peaks_x]
# 				for x in peaks_x:
# 					axs[i].annotate(str(x), xy=(x, 1.05 * df.iloc[x]))
# 				axs[i].plot(df, linewidth=0.25, color=channels_of_interest[ch])
# 				axs[i].set_title(ch, fontdict={'fontsize': 8, 'fontweight': 'medium'})
# 				axs[i].set_xlim([x_window_start, x_window_end])
# 				axs[i].set_ylabel('RFU', fontsize=6)
# 				axs[i].set_xlabel('Fragment Size', fontsize=6)
# 				axs[i].yaxis.set_tick_params(labelsize=6)
# 				y_max = df[x_window_start:x_window_end].max()
# 				y_top = y_max + 0.1 * abs(y_max)
# 				if y_top < 200:
# 					y_top = 200
# 				y_min = df[x_window_start:x_window_end].min()
# 				y_bot = y_min - 0.1 * abs(y_min)
# 				axs[i].set_ylim(top=y_top, bottom=y_bot)

# 			pdf.savefig()
# 			plt.close(p)

# 			ch_4 = re.sub(r'channel_\d', 'channel_4', channel)
# 			ch_repeat_4 = re.sub(r'channel_\d', 'channel_4', channel_repeat)

# 			p, axs = plt.subplots(nrows=2, ncols=1)
# 			p.subplots_adjust(hspace=0.5)
# 			p.suptitle(case.name)

# 			for i, ch in enumerate([ch_4, ch_repeat_4]):
# 				df = case.df[ch]
# 				axs[i].plot(df, linewidth=0.25)
# 				axs[i].plot(df[case.ladder[ch]], 'x', color='red')
# 				axs[i].set_title(ch, fontdict={'fontsize': 8, 'fontweight': 'medium'})
# 				axs[i].set_ylabel('RFU', fontsize=6)
# 				axs[i].set_xlabel('Fragment Size', fontsize=6)
# 				axs[i].yaxis.set_tick_params(labelsize=6)
# 				axs[i].set_ylim(top=2000)
# 				axs[i].set_xlim(left=1000)

# 			pdf.savefig()
# 			plt.close(p)

# 		for ch in scl:

# 			p, axs = plt.subplots(nrows=1, ncols=1)
# 			p.subplots_adjust(hspace=0.5)
# 			p.suptitle(case.name)

# 			axs.set_title(ch, fontdict={'fontsize': 8, 'fontweight': 'medium'})
# 			axs.set_ylabel('RFU', fontsize=6)
# 			axs.set_xlabel('Fragment Size', fontsize=6)
# 			axs.yaxis.set_tick_params(labelsize=6)

# 			if case.ladder_success: clr = 'green'
# 			else: clr = 'red'
# 			axs.plot(case.ladder_SCL, case.df[ch][case.ladder_SCL], 'o', fillstyle='none', color=clr)
# 			axs.plot(case.df[ch], linewidth=0.25, color=channels_of_interest[ch])
# 			axs.plot(case.df['decay'], linewidth=0.25, color=clr)

# 			pdf.savefig()
# 			plt.close(p)
# 		print('Done making {}'.format(multipage))

def pick_peak_one(case):
	case.ladder_success = False
	scldf = case.df['SCL_channel_1']
	#Goal is to return the farther (on x axis) of the two tallest peaks
	mask = scldf.index.isin(range(1500,2300))	# this range was determined by looking at 250+ cases
	min_dist=20
	if mask.size == scldf.size:
		peaks_x, _ = find_peaks(scldf.where(mask, 0), distance=min_dist)
		peaks_2tallest = sorted([(x, scldf[x]) for x in peaks_x], key=lambda coor: coor[1], reverse=True)[:2]
		peak_farther_of_2tallest = sorted(peaks_2tallest, key=lambda coor: coor[0], reverse=True)[0]
		case.peak_one = peak_farther_of_2tallest
		mask = scldf.index.isin(range(case.peak_one[0], scldf.size))
		peaks_x, _ = find_peaks(scldf.where(mask, 0), distance=min_dist)
		case.peaks = [(x, scldf[x]) for x in sorted(peaks_x, reverse=False)]
	else:
		print('\tSkipping {} due to size mismatch, likely due to multiple files being added to the same column in the case DataFrame column'.format(case.name))
		for f in case.files['SCL']:
			print('\t\t{}'.format(f))
	return case

def make_decay_curve(case):
	a = case.peak_one[1]
	b = 0.5
	x_decay = np.array(range(case.peak_one[0],len(case.df.index.tolist())))
	i = 0
	while i < 20:
		i += 0.1
		y_decay = a*b**(i*(x_decay - case.peak_one[0])/case.peak_one[0])
		decay = pd.Series(data=y_decay,index=x_decay)
		decay.name = 'decay'
		if decay.name not in case.df.columns: case.df = pd.concat([case.df, decay], axis=1, sort=False)
		else: case.df[decay.name] = decay
		case = evaluate_SCL(case, decay)
		if case.residual <= 10:
			case.ladder_success = True
			break
	case.decay_value = i
	return case

def evaluate_SCL(case, decay):
	qualifying_peaks = [(x,y) for x,y in case.peaks if y > decay[x]]
	combos = [list(c) for c in combinations(qualifying_peaks, 3)]
	combos.sort(key=lambda coor:coor[0])
	case.ladder_SCL = [400,100,300,200]	# just some made up ladder
	case.residual = 1000000
	for combo in combos:
		ladder_SCL = [case.peak_one[0]] + [x for x,y in combo]
		poly_current, res_current, rank, singular_values, rcond = np.polyfit(ladder_SCL, [100,200,300,400], 1, full=True)
		res_current = res_current[0]
		if res_current < case.residual:
			case.residual = res_current
			case.ladder_SCL = ladder_SCL
	return case

def build_ladder(df, size_standard, label_name):
	choices, std = reduce_choices(df, label_name)
	ss = np.array(size_standard)
	if len(choices) < len(size_standard):
		print('\tWARNING: len(choices) = {}, k = {}'.format(len(choices), len(size_standard)))
	X = np.array([sorted(list(c)) for c in combinations(choices, len(size_standard))])
	# print('\t{} choose {} -> {:,} combos'.format(len(choices), len(size_standard), len(X)))
	pfit_zx = np.polyfit(ss, X.T, deg=1, full=True)
	residuals_zx = pfit_zx[1]
	X_mean = np.expand_dims(np.mean(X,axis=1),axis=1)
	R_sq_zx = 1.0 - (np.square(residuals_zx) / np.sum(np.square(X - X_mean)))
	# i = np.argmax(R_sq_zx)
	ranked_R_sq, indices = np.unique(R_sq_zx, return_index=True)
	indices = indices.tolist()
	indices.reverse()
	for i in indices:
		ladder = X[i]
		Y = df[ladder]
		# print('len(ladder) = {}'.format(len(ladder)))
		Ygrubb = grubbs.test(Y.tolist(), alpha=0.05)
		if len(Y) == len(Ygrubb):
			break
	return ladder

problem_cases = ['19KD-348M0008_TCRB-B_channel_4_repeat']

def reduce_choices(df, label_name):
	t = 2.0
	peaks_x_restricted, _ = find_peaks(df, height=[20,1000], distance=30, width=2)
	peaks_x, _ = find_peaks(df)
	coor = [(x,df[x]) for x in peaks_x]
	tallest = sorted(coor, key=lambda x: x[1])[-1]
	choices_x = [x for x in peaks_x_restricted if x > tallest[0]]
	choices_y = [df[x] for x in choices_x]
	# choices_y_grubbs = grubbs.test(choices_y, alpha=0.05)
	# choices_x_reduced = [x for x in choices_x if df[x] in choices_y_grubbs]
	polyxy = np.poly1d(np.polyfit(choices_x, choices_y, deg=1))
	polybaseline = np.poly1d(np.polyfit(df.index.tolist()[choices_x[0]:], df[choices_x[0]:],deg=1))
	std = np.std(choices_y)
	std2_below = polyxy(df.index.to_list()) - t*std
	std2_above = polyxy(df.index.to_list()) + t*std
	std2 = [(x1,x2) for x1, x2 in zip(std2_below, std2_above)]
	peaks_x, _ = find_peaks(df, height=[std2_below, std2_above], prominence=20, width=2)
	choices_x = [x for x in peaks_x if x > tallest[0]]

	if len(choices_x) > 20:
		p, axs = plt.subplots(nrows=1, ncols=1)
		axs.plot(df[choices_x], 'o', fillstyle='none')
		# axs[0].plot(df[best_ladder_i], 'x', fillstyle='none')
		axs.plot(df.index.tolist(), df, linewidth=0.25)
		# axs[0].plot(polyzx(ss),np.zeros(len(ss)), '|', color='red')
		# axs[0].plot(best_ladder_i, polyxy(best_ladder_i), linewidth=0.25)
		axs.plot(df.index.tolist(), polyxy(df.index.tolist()) + t*std, linewidth=0.25)
		axs.plot(df.index.tolist(), polyxy(df.index.tolist()) - t*std, linewidth=0.25)
		plt.ylim((-500,1000))
		# plt.show()
		plt.close(p)
	return choices_x, std

def size_standard(case, channel='channel_4'):
	rox500_16 = [35, 50, 75, 100, 139, 150, 160, 200, 250, 300, 340, 350, 400, 450, 490, 500]
	rox500_14 = [35, 50, 75, 100, 139, 150, 160, 200, 250, 300, 340, 350, 400, 450]
	rox500_13 = [50, 75, 100, 139, 150, 160, 200, 250, 300, 340, 350, 400, 450]
	rox500_75_400 = [75, 100, 139, 150, 160, 200, 250, 300, 340, 350, 400]
	rox500 = rox500_75_400
	case.rox500 = rox500[:]
	ladder_channels = [ch for ch in case.df.columns if channel in ch]
	for ch in ladder_channels:
		label_name = '_'.join([case.name, ch])
		case.ladder[ch] = build_ladder(case.df[ch], rox500, label_name)
	return case

def baseline_correction(case):
	for ch in case.df.columns:
		# ch_repeat = '_'.join([ch, 'repeat'])
		if ch in channels_of_interest.keys() and 'SCL' not in ch:
			label_name = '_'.join([case.name, ch])
			if debug: print(label_name)
			for i in range(0,3):
				_, prop = find_peaks(case.df[ch], prominence=1, distance=10)
				bases = sorted(list(set(np.concatenate([prop['left_bases'], prop['right_bases']]))))
				spl = InterpolatedUnivariateSpline(bases, case.df[ch][bases])
				spl_df = pd.Series(spl(case.df.index.tolist()))
				case.df[ch] = case.df[ch] - spl_df
	return case

channels_of_interest = {
			'IGH-A_channel_1':'blue',
			'IGH-B_channel_1':'blue',
			'IGH-C_channel_2':'green',
			'IGK-A_channel_1':'blue',
			'IGK-B_channel_1':'blue',
			'TCRB-A_channel_1':'blue',
			'TCRB-A_channel_2':'green',
			'TCRB-B_channel_1':'blue',
			'TCRB-C_channel_1':'blue',
			'TCRB-C_channel_2':'green',
			'TCRG-A_channel_1':'blue',
			'TCRG-A_channel_2':'green',
			'TCRG-B_channel_1':'blue',
			'TCRG-B_channel_2':'green',
			'SCL_channel_1':'black',
			'IGH-A_channel_1_repeat':'blue',
			'IGH-B_channel_1_repeat':'blue',
			'IGH-C_channel_2_repeat':'green',
			'IGK-A_channel_1_repeat':'blue',
			'IGK-B_channel_1_repeat':'blue',
			'TCRB-A_channel_1_repeat':'blue',
			'TCRB-A_channel_2_repeat':'green',
			'TCRB-B_channel_1_repeat':'blue',
			'TCRB-C_channel_1_repeat':'blue',
			'TCRB-C_channel_2_repeat':'green',
			'TCRG-A_channel_1_repeat':'blue',
			'TCRG-A_channel_2_repeat':'green',
			'TCRG-B_channel_1_repeat':'blue',
			'TCRG-B_channel_2_repeat':'green',
			'SCL_channel_1_repeat':'black'
	}
regions_of_interest = {
			'IGH-A_channel_1':[(310,360)],
			'IGH-B_channel_1':[(250,295)],
			'IGH-C_channel_2':[(100,170)],
			'IGK-A_channel_1':[(120,160),(190,210),(260,300)],
			'IGK-B_channel_1':[(210,250),(270,300),(350,390)],
			'TCRB-A_channel_1':[(240,285)],
			'TCRB-A_channel_2':[(240,285)],
			'TCRB-B_channel_1':[(240,285)],
			'TCRB-C_channel_1':[(170,210),(285,325)],
			'TCRB-C_channel_2':[(170,210),(285,325)],
			'TCRG-A_channel_1':[(175,195),(230,255)],
			'TCRG-A_channel_2':[(145,175),(195,230)],
			'TCRG-B_channel_1':[(110,140),(195,220)],
			'TCRG-B_channel_2':[(80,110),(160,195)],
			'IGH-A_channel_1_repeat':[(310,360)],
			'IGH-B_channel_1_repeat':[(250,295)],
			'IGH-C_channel_2_repeat':[(100,170)],
			'IGK-A_channel_1_repeat':[(120,160),(190,210),(260,300)],
			'IGK-B_channel_1_repeat':[(210,250),(270,300),(350,390)],
			'TCRB-A_channel_1_repeat':[(240,285)],
			'TCRB-A_channel_2_repeat':[(240,285)],
			'TCRB-B_channel_1_repeat':[(240,285)],
			'TCRB-C_channel_1_repeat':[(170,210),(285,325)],
			'TCRB-C_channel_2_repeat':[(170,210),(285,325)],
			'TCRG-A_channel_1_repeat':[(175,195),(230,255)],
			'TCRG-A_channel_2_repeat':[(145,175),(195,230)],
			'TCRG-B_channel_1_repeat':[(110,140),(195,220)],
			'TCRG-B_channel_2_repeat':[(80,110),(160,195)]
	}

def index_of_peaks_to_annotate(case):
	for ch in case.df.columns:
		x_col_name = 'x_fitted_' + re.sub(r'channel_\d','channel_4', ch)
		if ch in regions_of_interest.keys():
			peaks_x, _ = find_peaks(case.df[ch], height=600, prominence=100)
			peaks_in_roi = []
			for x_start, x_end in regions_of_interest[ch]:
				peaks_in_roi.extend([x for x in peaks_x if case.df[x_col_name][x] >= x_start and case.df[x_col_name][x] <= x_end])
			peaks_y = case.df[ch][peaks_in_roi].to_list()
			peaks_in_roi = [x for y,x in sorted(zip(peaks_y, peaks_in_roi), reverse=True)]
			if len(peaks_in_roi) > 5:
				peaks_in_roi = peaks_in_roi[0:5]


			case.index_of_peaks_to_annotate[ch] = peaks_in_roi[:]
	return case

def plot_scl(case, ch, plot_dict, w, h):
	if ch in channels_of_interest.keys() and 'SCL' in ch:
		TOOLTIPS = [("(x,y)", "($x{1.1}, $y{int})")]
		x_col_name = 'x_fitted_' + re.sub(r'channel_\d','channel_4', ch)
		x = case.df[ch].index.to_list()
		y = case.df[ch].to_list()
		p = figure(tools='pan,wheel_zoom,reset',title=ch, x_axis_label='fragment size', y_axis_label='RFU', width=w, height=h, x_range=(1000, max(x)), tooltips=TOOLTIPS)
		p.line(x, y, line_width=0.5, color=channels_of_interest[ch])
		plot_dict[ch] = p
	return plot_dict

def plot_channels_of_interest(case, ch, plot_dict, w, h):
	if ch in channels_of_interest.keys() and 'SCL' not in ch:
		TOOLTIPS = [("(x,y)", "($x{1.1}, $y{int})")]
		x_col_name = 'x_fitted_' + re.sub(r'channel_\d','channel_4', ch)
		p = figure(tools='pan,wheel_zoom,reset',title=ch, x_axis_label='fragment size', y_axis_label='RFU', width=w, height=h, x_range=(75,400), tooltips=TOOLTIPS)
		x = case.df[x_col_name].to_list()
		y = case.df[ch].to_list()
		p.line(x, y, line_width=0.5, color=channels_of_interest[ch])
		if ch in regions_of_interest.keys():
			# mark regions in gray
			for x_left, x_right in regions_of_interest[ch]:
				roi = BoxAnnotation(left=x_left, right=x_right, fill_color='black', fill_alpha=0.05)
				p.add_layout(roi)
		plot_dict[ch] = p
	return plot_dict

def plot_peaks_of_interest(case, ch, plot_dict, w, h):
	if ch in regions_of_interest.keys():
		x_col_name = 'x_fitted_' + re.sub(r'channel_\d','channel_4', ch)
		p = plot_dict[ch]
		peaks_index = case.index_of_peaks_to_annotate[ch]
		x_peaks = case.df[x_col_name][peaks_index].to_list()
		y_peaks = case.df[ch][peaks_index].to_list()
		if len(y_peaks) > 0:
			p.y_range.end = 1.3*max(y_peaks)
		else:
			p.y_range.end = 1000
		for x,y in zip(x_peaks, y_peaks):
			mytext = Label(angle=1, x=x, y=int(y), text='{:.1f}'.format(x), x_offset=0, y_offset=2, text_font_size='8pt')
			p.add_layout(mytext)
	return plot_dict

def plot_size_standard(case, ch, plot_dict, w, h):
	if ch in channels_of_interest.keys() and 'SCL' not in ch:
		TOOLTIPS = [("(x,y)", "($x{1.1}, $y{int})")]
		ch_4 = re.sub(r'channel_\d', 'channel_4', ch)
		case.df[ch_4].index.rename('x')
		x = case.df[ch_4].index.to_list()
		y = case.df[ch_4].to_list()
		x_ladder = case.ladder[ch_4]
		y_ladder = case.df[ch_4][x_ladder].to_list()
		p = figure(tools='pan,wheel_zoom,reset',title=ch_4, x_axis_label='fragment size', y_axis_label='RFU', width=w, height=int(h/2.0), x_range=(1000, max(x)), y_range = (-200, max(y_ladder)+200), tooltips=TOOLTIPS)
		p.line(x, y, line_width=0.5, color='red')
		p.ygrid.visible = False
		p.x(x_ladder, y_ladder)
		# print('x={}, y={}'.format(x_ladder, y_ladder))
		for x,y,label in zip(x_ladder, y_ladder, case.rox500):
			mytext = Label(angle=1, x=x, y=y, text=str(label), x_offset=0, y_offset=2, text_font_size='8pt')
			p.add_layout(mytext)
		plot_dict[ch_4] = p
	return plot_dict

def sync_axes(plot_dict):
	for ch, p in plot_dict.items():
		ch_repeat = ch + '_repeat'
		if ch_repeat in plot_dict.keys():
			if p.y_range.end >= plot_dict[ch_repeat].y_range.end:
				plot_dict[ch_repeat].x_range = p.x_range
				plot_dict[ch_repeat].y_range = p.y_range
			else:
				p.x_range = plot_dict[ch_repeat].x_range
				p.y_range = plot_dict[ch_repeat].y_range
	return plot_dict

def plot_case(case, w=1000, h=300):
	# Need to break this in to sub functions. It's hard to follow like this.
	silence(FIXED_SIZING_MODE, True)
	TOOLTIPS = [("(x,y)", "($x{1.1}, $y{int})")]
	output_file(case.name + '.html')
	plot_dict = {}
	for ch in sorted(case.df.columns):
		plot_dict = plot_scl(case, ch, plot_dict, w, h)
		plot_dict = plot_channels_of_interest(case, ch, plot_dict, w, h)
		plot_dict = plot_size_standard(case, ch, plot_dict, w, h)
		plot_peaks_of_interest(case, ch, plot_dict, w, h)

	plot_dict = sync_axes(plot_dict)

	plot_keys = sorted([key for key in plot_dict.keys() if 'SCL' not in key])
	scl_keys = sorted([key for key in plot_dict.keys() if 'SCL' in key])
	plot_keys = [*scl_keys, *plot_keys]
	plots = column([plot_dict[ch] for ch in plot_keys], sizing_mode='fixed')

	show(plots)
	save(plots)

# def reindex_case(case):
# 	for ch in case.df.columns:
# 		if ch in channels_of_interest.keys():
# 			i_dict = {}
# 			if 'channel' in ch and not ch.startswith('x_fitted'):
# 				x_col_name = 'x_fitted_' + re.sub(r'channel_\d','channel_4', ch)
# 				X = np.around(10 * case.df[x_col_name])
# 				# print('X = {}'.format(X))
# 				i_dict = {int(x): set() for x in X}
# 				for i, x in X.items():
# 					x = int(x)
# 					i_dict[x].add(case.df[ch][i])
# 				i_dict = {x:max(v) for x,v in i_dict.items() if x >= 0}
# 				case.re_df[ch] = pd.Series(i_dict)
# 				# case.re_df[ch] = pd.Series(index=X, data=case.df[ch])
# 				# print(case.re_df[ch])
# 	return case

debug = False
use_timestamp = True

def main():
	owd = os.getcwd()	# original working directory
	path = os.path.abspath(sys.argv[1])
	os.chdir(path)
	cases = organize_files(path)
	# output_path = os.path.join(path, '/plots')
	# if not os.path.exists(output_path): os.mkdir(output_path)
	for case_name, case in cases.items():
		# if '0084' in case_name:
		print('Processing raw data for {}'.format(case_name))
		case = gather_case_data(case, case_name, path)
		case = size_standard(case)
		case = baseline_correction(case)
		case = pick_peak_one(case)
		case = make_decay_curve(case)
		case = local_southern(case)
		# case = reindex_case(case)
		case = index_of_peaks_to_annotate(case)
		# plot_case_pdf(case)
		plot_case(case, w=1100, h=350)

if __name__ == '__main__':
	main()