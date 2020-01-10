#!/usr/bin/env python3

# Importing Packages
import os
import sys
import re
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from itertools import combinations
from outliers import smirnov_grubbs as grubbs

from bokeh.io import output_file, show, save
from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.models import BoxAnnotation, Label, Range1d, WheelZoomTool, ResetTool, PanTool, LegendItem, Legend
from bokeh.core.validation.warnings import FIXED_SIZING_MODE
from bokeh.core.validation import silence

TOOLTIPS = [("(x,y)", "($x{1.1}, $y{int})")]
silence(FIXED_SIZING_MODE, True)

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
			'TCRB-C_channel_3':'orange',
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

roi_clonality = {
			'IGH-A_channel_1':[(310,360,'FR1-JH','blue')],
			'IGH-B_channel_1':[(250,295,'FR2-JH','blue')],
			'IGH-C_channel_2':[(100,170,'FR3-JH','blue')],
			'IGK-A_channel_1':[(120,160,'Vκ-Jκ-1','blue'),(190,210,'Vκ-Jκ-2','green'),(260,300,'Vκ-Jκ-3','red')],
			'IGK-B_channel_1':[(210,250,'Vκ-Kde-1','blue'),(270,300,'Vκ-Kde-2','green'),(350,390,'Vκ-Kde-3','red')],
			'TCRB-A_channel_1':[(240,285,'Vβ_Jβ_Jβ2.X','blue')],
			'TCRB-A_channel_2':[(240,285,'Vβ_Jβ_Jβ1.X','blue')],
			'TCRB-B_channel_1':[(240,285,'Vβ_Jβ2','blue')],
			'TCRB-C_channel_1':[(170,210,'Dβ_Jβ_Dβ2','blue'),(285,325,'Dβ_Jβ_Dβ1','green')],
			'TCRB-C_channel_2':[(170,210,'Dβ_Jβ_Dβ2','blue'),(285,325,'Dβ_Jβ_Dβ1','green')],
			'TCRG-A_channel_1':[(175,195,'Vγ10_Jγ1.1_2.1','blue'),(230,255,'Vγ1-8_Jγ1.1_2.1','green')],
			'TCRG-A_channel_2':[(145,175,'Vγ10_Jγ1.3_2.3','blue'),(195,230,'Vγ1-8_Jγ1.3_2.3','green')],
			'TCRG-B_channel_1':[(110,140,'Vγ11_Jγ1.1_2.1','blue'),(195,220,'Vγ9_Jγ1.1_2.1','green')],
			'TCRG-B_channel_2':[(80,110,'Vγ11_Jγ2.1_2.3','blue'),(160,195,'Vγ9_Jγ1.3_2.3','green')],
			'IGH-A_channel_1_repeat':[(310,360,'FR1-JH','blue')],
			'IGH-B_channel_1_repeat':[(250,295,'FR2-JH','blue')],
			'IGH-C_channel_2_repeat':[(100,170,'FR3-JH','blue')],
			'IGK-A_channel_1_repeat':[(120,160,'Vκ-Jκ-1','blue'),(190,210,'Vκ-Jκ-2','green'),(260,300,'Vκ-Jκ-3','red')],
			'IGK-B_channel_1_repeat':[(210,250,'Vκ-Kde-1','blue'),(270,300,'Vκ-Kde-2','green'),(350,390,'Vκ-Kde-3','red')],
			'TCRB-A_channel_1_repeat':[(240,285,'Vβ_Jβ_Jβ2.X','blue')],
			'TCRB-A_channel_2_repeat':[(240,285,'Vβ_Jβ_Jβ1.X','blue')],
			'TCRB-B_channel_1_repeat':[(240,285,'Vβ_Jβ2','blue')],
			'TCRB-C_channel_1_repeat':[(170,210,'Dβ_Jβ_Dβ2','blue'),(285,325,'Dβ_Jβ_Dβ1','green')],
			'TCRB-C_channel_2_repeat':[(170,210,'Dβ_Jβ_Dβ2','blue'),(285,325,'Dβ_Jβ_Dβ1','green')],
			'TCRG-A_channel_1_repeat':[(175,195,'Vγ10_Jγ1.1_2.1','blue'),(230,255,'Vγ1-8_Jγ1.1_2.1','green')],
			'TCRG-A_channel_2_repeat':[(145,175,'Vγ10_Jγ1.3_2.3','blue'),(195,230,'Vγ1-8_Jγ1.3_2.3','green')],
			'TCRG-B_channel_1_repeat':[(110,140,'Vγ11_Jγ1.1_2.1','blue'),(195,220,'Vγ9_Jγ1.1_2.1','green')],
			'TCRG-B_channel_2_repeat':[(80,110,'Vγ11_Jγ2.1_2.3','blue'),(160,195,'Vγ9_Jγ1.3_2.3','green')],
}

channel_colors = {
	'channel_1':'blue',
	'channel_2':'green',
	'channel_3':'purple',
	'channel_4':'red',
	'channel_5':'darkgoldenrod',
	'SCL':'black'
}

def pretty_name(c,t):
	if 'channel' in c:
		channel = re.findall(r'channel_\d', c)[0]
		if 'repeat' in c:
			pc = '_'.join([t, channel, 'repeat'])
		else:
			pc = '_'.join([t, channel])
	else:
		pc = c
	return pc

def organize_clonality_files(path):
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
	cd = {case_name : { t : [f for f in csv_list if case_name in f and t in f] for t in tests } for case_name in case_names}
	cases = {case_name: Case() for case_name in case_names}
	for case_name,c in cases.items():
		c.name = case_name
		c.files = cd[case_name]
		# c.ladder = {}
		# c.rox500 = []
		# c.index_of_peaks_to_annotate = {}
		# c.index_of_artifactual_peaks = {}
		# c.index_of_replicate_peaks = {}
		# c.allelic_ladder = None
		# c.plot_labels = {}
	return cases

class Case(object):
	"""	I'm sure there's a better way than making a dummy class like this.
	"""
	def __init__(self):
		self.name = None
		self.files = {}
		self.ladder = {}
		self.rox500 = []
		self.index_of_peaks_to_annotate = {}
		self.index_of_artifactual_peaks = {}
		self.index_of_replicate_peaks = {}
		self.allelic_ladder = None
		self.plot_labels = {}
		self.widths = {}
		self.abberant_peaks = {}
		self.some_peaks = {}
		self.some_upside_down_peaks = {}
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
	for ch_ss, ladder in case.ladder.items():
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
		col_name = '_'.join(['x_fitted', ch_ss])
		x_df.columns = [col_name]
		case.df = pd.concat([case.df, x_df], axis=1, sort=False)
	return case

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

def reduce_choices(ds, label_name):
	t = 2.0
	# print(ds)
	peaks_x_restricted, _ = find_peaks(ds, height=[20,1000], distance=30, width=2)
	peaks_x, _ = find_peaks(ds)
	coor = [(x,ds[x]) for x in peaks_x]
	# print('label_name = {}'.format(label_name))
	# print('coor = {}'.format(coor))
	tallest = sorted(coor, key=lambda x: x[1])[-1]
	choices_x = [x for x in peaks_x_restricted if x > tallest[0]]
	choices_y = [ds[x] for x in choices_x]
	# choices_y_grubbs = grubbs.test(choices_y, alpha=0.05)
	# choices_x_reduced = [x for x in choices_x if ds[x] in choices_y_grubbs]
	polyxy = np.poly1d(np.polyfit(choices_x, choices_y, deg=1))
	# polybaseline = np.poly1d(np.polyfit(ds.index.tolist()[choices_x[0]:], ds[choices_x[0]:],deg=1))
	std = np.std(choices_y)
	std2_below = polyxy(ds.index.to_list()) - t*std
	std2_above = polyxy(ds.index.to_list()) + t*std
	# std2 = [(x1,x2) for x1, x2 in zip(std2_below, std2_above)]
	peaks_x, _ = find_peaks(ds, height=[std2_below, std2_above], prominence=20, width=2)
	choices_x = [x for x in peaks_x if x > tallest[0]]
	return choices_x, std

def size_standard(case, ch_ss_num=4):
	rox500_16 = [35, 50, 75, 100, 139, 150, 160, 200, 250, 300, 340, 350, 400, 450, 490, 500]
	rox500_14 = [35, 50, 75, 100, 139, 150, 160, 200, 250, 300, 340, 350, 400, 450]
	rox500_13 = [50, 75, 100, 139, 150, 160, 200, 250, 300, 340, 350, 400, 450]
	rox500_75_400 = [75, 100, 139, 150, 160, 200, 250, 300, 340, 350, 400]
	rox500_75_450 = [75, 100, 139, 150, 160, 200, 250, 300, 340, 350, 400, 450]
	rox500 = rox500_75_400
	case.rox500 = rox500[:]
	ch_ss = 'channel_'+str(ch_ss_num)
	ladder_channels = [ch for ch in case.df.columns if ch_ss in ch and 'x_fitted' not in ch]
	# print('ladder_channels = {}'.format(ladder_channels))
	for ch in ladder_channels:
		label_name = '_'.join([case.name, ch])
		case.ladder[ch] = build_ladder(case.df[ch], rox500, label_name)
	return case

def baseline_correction_simple(case, ch_list=None, ch_ss_num=4):
	if ch_list is None:
		ch_list = case.df.columns.to_list()
	else:
		ch_list = list(set(case.df.columns.to_list()) & set(ch_list))
	ch_ss = 'channel_' + str(ch_ss_num)
	ch_list = [ch for ch in ch_list if ch_ss not in ch]
	for ch in ch_list:
		peaks_i, props = find_peaks(case.df[ch], prominence=50)
		I = case.df.index.to_list()
		I_1k = I[1000:]
		# right_bases = props['right_bases']
		# left_bases = props['left_bases']
		# I_exclude = set()
		# for l,r in zip(left_bases, right_bases):
		# 	I_exclude.update(set(range(l,r)))
		# I = [i for i in I if i not in I_exclude]
		x_baseline = case.df[ch][I_1k].to_list()
		# x_avg = mean(x_baseline)
		polyxy = np.poly1d(np.polyfit(I_1k, x_baseline, deg=1))
		case.df[ch] = case.df[ch] - polyxy(case.df.index.to_list())
	# case.df = case.df.where(case.df > 0, 0)
	return case

def baseline_correction_upside_down(case, ch_list=None, ch_ss_num=4, iterations=3, prominence=1, distance=20):
	if ch_list is None:
		ch_list = case.df.columns.to_list()
	else:
		ch_list = list(set(case.df.columns.to_list()) & set(ch_list))
	ch_ss = 'channel_' + str(ch_ss_num)
	ch_list = [ch for ch in ch_list if ch_ss not in ch]
	for ch in ch_list:
		peaks_start, _ = find_peaks(case.df[ch], prominence=prominence, distance=distance)
		df = case.df[ch] * -1
		peaks_start, _ = find_peaks(df, prominence=prominence, distance=distance)
		all_your_base = set()
		for i in range(0,iterations):
			bases, props = find_peaks(df, prominence=prominence, distance=distance)
			spl = InterpolatedUnivariateSpline(bases, df[bases], bbox=[bases[0], bases[int(len(bases)/2)]])
			spl_df = pd.Series(spl(case.df.index.tolist()))
			df = df - spl_df
		case.df[ch] = df * -1
		peaks_finish, _ = find_peaks(case.df[ch], prominence=prominence, distance=distance)
		abberant_peaks = set(peaks_finish) - set(peaks_start)
		case.abberant_peaks[ch] = abberant_peaks
	return case

def baseline_correction_advanced(case, ch_list=None, ch_ss_num=4, iterations=3, prominence=1, distance=20):
	if ch_list is None:
		ch_list = case.df.columns.to_list()
	else:
		ch_list = list(set(case.df.columns.to_list()) & set(ch_list))
	ch_ss = 'channel_' + str(ch_ss_num)
	ch_list = [ch for ch in ch_list if ch_ss not in ch]
	for ch in ch_list:
		peaks_start, _ = find_peaks(case.df[ch], prominence=prominence, distance=distance)
		all_your_base = set()
		for i in range(0,iterations):
			peaks_current, props = find_peaks(case.df[ch], prominence=prominence, distance=distance)
			# abberant_peaks = set(peaks_current) - set(peaks_original)
			bases = set(np.concatenate([props['left_bases'], props['right_bases']]))
			all_your_base = all_your_base | bases
			# bases = bases | abberant_peaks
			bases = sorted(list(bases))
			# bases = sorted(list(set(np.concatenate([props['left_bases'], props['right_bases']]))))
			# bases = [b for b in bases if b >=0]
			# spl = InterpolatedUnivariateSpline(bases, case.df[ch][bases])
			# print('len(bases) = {}'.format(len(bases)))
			spl = InterpolatedUnivariateSpline(bases, case.df[ch][bases], ext=1)
			# spl = interp1d(bases, case.df[ch][bases], fill_value='extrapolate')
			spl_df = pd.Series(spl(case.df.index.tolist()))
			case.df[ch] = case.df[ch] - spl_df
		# peaks_finish, _ = find_peaks(case.df[ch], prominence=prominence, distance=distance)
		# abberant_peaks = set(peaks_finish) - set(peaks_start)
		# case.abberant_peaks[ch] = abberant_peaks
		# case.abberant_peaks[ch] = all_your_base
	return case

def index_of_peaks_to_annotate(case):
	for ch in case.df.columns:
		x_col_name = 'x_fitted_' + re.sub(r'channel_\d','channel_4', ch)
		if ch in roi_clonality.keys():
			peaks_x, _ = find_peaks(case.df[ch], prominence=100, height=300)
			peaks_in_all_roi = []
			for x_start, x_end, _, _ in roi_clonality[ch]:
				peaks_in_current_roi = [x for x in peaks_x if case.df[x_col_name][x] >= x_start and case.df[x_col_name][x] <= x_end]
				peaks_y = case.df[ch][peaks_in_current_roi].to_list()
				peaks_in_current_roi = [x for y,x in sorted(zip(peaks_y, peaks_in_current_roi), reverse=True)]
				if len(peaks_in_current_roi) > 5:
					peaks_in_all_roi.extend(peaks_in_current_roi[0:5])
				else:
					peaks_in_all_roi.extend(peaks_in_current_roi)
			case.index_of_peaks_to_annotate[ch] = peaks_in_all_roi[:]
	return case

def find_artifactual_peaks(case):
	for ch in case.df.columns:
		if 'channel_3' in ch and 'SCL' not in ch:
			ch_4 = re.sub(r'channel_\d', 'channel_4', ch)
			label_name = case.name + '_' + ch
			ladder = case.ladder[ch_4]
			peaks_temp, _ = find_peaks(case.df[ch], height=500)
			peaks_i = []
			for i in peaks_temp:
				if i >= ladder[0] and i <= ladder[-1]:
					peaks_i.append(i)
			case.index_of_artifactual_peaks[ch] = peaks_i[:]
	return case

def plot_scl(case, ch, plot_dict, w, h):
	if ch in channels_of_interest.keys() and 'SCL' in ch:
		ch_num = re.findall(r'channel_\d', ch)[0]
		label_name = case.name + '_' + ch
		x_col_name = 'x_fitted_' + re.sub(r'channel_\d','channel_4', ch)
		x = case.df[ch].index.to_list()
		y = case.df[ch].to_list()
		p = figure(tools='pan,wheel_zoom,reset',title=label_name, x_axis_label='fragment size', y_axis_label='RFU', width=w, height=h, x_range=(1000, max(x)), tooltips=TOOLTIPS)
		p.line(x, y, line_width=0.5, color=channel_colors.get(ch_num, 'blue'))
		plot_dict[ch] = p
	return plot_dict

def plot_channels_of_interest(case, ch, plot_dict, w, h, ch_ss_num=4):
	if ch in channels_of_interest.keys() and 'SCL' not in ch:
		ch_num = re.findall(r'channel_\d', ch)[0]
		label_name = case.name + '_' + ch
		x_col_name = 'x_fitted_' + re.sub(r'channel_\d','channel_'+str(ch_ss_num), ch)
		p = figure(tools='pan,wheel_zoom,reset',title=label_name, x_axis_label='fragment size', y_axis_label='RFU', width=w, height=h, x_range=(75,400), tooltips=TOOLTIPS)
		x = case.df[x_col_name].to_list()
		y = case.df[ch].to_list()
		p.line(x, y, line_width=0.5, color=channel_colors.get(ch_num, 'blue'))
		plot_dict[ch] = p
	return plot_dict

def highlight_roi_clonality(case, ch, plot_dict, w, h):
	if ch in roi_clonality.keys():
		p = plot_dict[ch]
		legends = []
		for x_left, x_right, roi_name, roi_color in roi_clonality[ch]:
			dummy_dot = p.line([0,0],[1,1], line_width=20, color=roi_color, alpha=0.10)
			roi = BoxAnnotation(left=x_left, right=x_right, fill_color=roi_color, fill_alpha=0.05)
			p.add_layout(roi)
			legends.append(LegendItem(label=roi_name, renderers=[dummy_dot]))
		p.add_layout(Legend(items=legends, location='top_right'))
		# print(p.legend.items)
		plot_dict[ch] = p
	return plot_dict

def plot_peaks_of_interest(case, ch, plot_dict, w, h, replicate_only, ch_ss_num=4):
	if ch in roi_clonality.keys():
		x_col_name = 'x_fitted_' + re.sub(r'channel_\d','channel_'+str(ch_ss_num), ch)
		p = plot_dict[ch]
		if replicate_only:
			peaks_index = case.index_of_replicate_peaks[ch]
		else:
			peaks_index = case.index_of_peaks_to_annotate[ch]
		x_peaks = case.df[x_col_name][peaks_index].to_list()
		y_peaks = case.df[ch][peaks_index].to_list()
		p.y_range.start = -100
		if len(y_peaks) > 0:
			p.y_range.end = 1.3*max(y_peaks)
		else:
			p.y_range.end = 1000
		for x,y in zip(x_peaks, y_peaks):
			mytext = Label(angle=1, x=x, y=int(y), text='{:.1f}'.format(x), x_offset=0, y_offset=2, text_font_size='8pt')
			p.add_layout(mytext)
	return plot_dict

def plot_size_standard(case, ch, plot_dict, w, h, ch_ss_num=4):
	# if ch in channels_of_interest.keys() and 'SCL' not in ch:
	ch_ss = re.sub(r'channel_\d', 'channel_' + str(ch_ss_num), ch)
	ch_num = re.findall(r'channel_\d', ch)[0]
	if ch_ss in case.ladder.keys():
		label_name = case.name + '_' + ch_ss
		# case.df[ch_ss].index.rename('x')
		x = case.df[ch_ss].index.to_list()
		y = case.df[ch_ss].to_list()
		x_ladder = case.ladder[ch_ss]
		y_ladder = case.df[ch_ss][x_ladder].to_list()
		p = figure(tools='pan,wheel_zoom,reset',title=label_name, x_axis_label='size standard', y_axis_label='RFU', width=w, height=int(h/2.0), x_range=(0, max(x)), y_range = (-200, max(y_ladder)+200), tooltips=TOOLTIPS)
		p.line(x, y, line_width=0.5, color=channel_colors.get(ch_num, 'blue'))
		p.ygrid.visible = False
		p.x(x_ladder, y_ladder)
		for x,y,label in zip(x_ladder, y_ladder, case.rox500):
			mytext = Label(angle=1, x=x, y=y, text=str(label), x_offset=0, y_offset=2, text_font_size='8pt')
			p.add_layout(mytext)
		plot_dict[ch_ss] = p
	return plot_dict

def plot_empty_channel_3(case, ch, plot_dict, w, h):
	if ch in channels_of_interest.keys() and 'SCL' not in ch:
		ch_3 = re.sub(r'channel_\d', 'channel_3', ch)
		label_name = case.name + '_' + ch_3
		x = case.df[ch_3].index.to_list()
		y = case.df[ch_3].to_list()
		x_ladder = case.index_of_artifactual_peaks[ch_3]
		y_ladder = case.df[ch_3][x_ladder].to_list()
		if len(y_ladder) > 0:
			p = figure(tools='pan,wheel_zoom,reset',title=label_name, x_axis_label='channel of artifactual peaks', y_axis_label='RFU', width=w, height=int(h/2.0), x_range=(0, max(x)), y_range = (-200, 1.5*max(y_ladder)), tooltips=TOOLTIPS)
		else:
			p = figure(tools='pan,wheel_zoom,reset',title=label_name, x_axis_label='channel of artifactual peaks', y_axis_label='RFU', width=w, height=int(h/2.0), x_range=(0, max(x)), tooltips=TOOLTIPS)
		p.line(x, y, line_width=0.5, color=channel_colors.get('channel_3', 'blue'))
		p.ygrid.visible = False
		# p.x(x_ladder, y_ladder)
		x_col_name = 'x_fitted_' + re.sub(r'channel_\d','channel_4', ch)
		x_fitted = case.df[x_col_name][x_ladder].to_list()
		for x,y,label in zip(x_ladder, y_ladder, x_fitted):
			mytext = Label(angle=1, x=x, y=y, text='{:.1f}'.format(label), x_offset=0, y_offset=2, text_font_size='8pt')
			p.add_layout(mytext)
		plot_dict[ch_3] = p
	return plot_dict

def sync_axes(plot_dict):
	sorted_keys = sorted(plot_dict.keys())
	p1 = plot_dict[sorted_keys[0]]
	p1.toolbar.active_scroll = p1.select_one(WheelZoomTool)
	for ch, p in plot_dict.items():
		p.tools = p1.tools
		p.toolbar.logo = None
		ch_repeat = ch + '_repeat'
		if ch_repeat in plot_dict.keys():
			if p.y_range.end is not None and plot_dict[ch_repeat].y_range.end is not None:
				if p.y_range.end >= plot_dict[ch_repeat].y_range.end:
					plot_dict[ch_repeat].x_range = p.x_range
					plot_dict[ch_repeat].y_range = p.y_range
				else:
					p.x_range = plot_dict[ch_repeat].x_range
					p.y_range = plot_dict[ch_repeat].y_range
	return plot_dict

def plot_clonality_case(case, replicate_only, w=1100, h=350):
	silence(FIXED_SIZING_MODE, True)
	plot_dict = {}
	for ch in sorted(case.df.columns):
		plot_dict = plot_scl(case, ch, plot_dict, w, h)
		plot_dict = plot_channels_of_interest(case, ch, plot_dict, w, h)
		plot_dict = highlight_roi_clonality(case, ch, plot_dict, w, h)
		plot_dict = plot_empty_channel_3(case, ch, plot_dict, w, h)
		plot_dict = plot_size_standard(case, ch, plot_dict, w, h)
		plot_dict = plot_peaks_of_interest(case, ch, plot_dict, w, h, replicate_only)

	plot_dict = sync_axes(plot_dict)

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

debug = False

def replicate_peaks(case):
	for ch in case.index_of_peaks_to_annotate.keys():
		if ch not in case.index_of_replicate_peaks.keys():
			case.index_of_replicate_peaks[ch] = []
			if 'repeat' not in ch:
				x_ch = 'x_fitted_' + re.sub(r'channel_\d','channel_4', ch)
				ch_repeat = ch + '_repeat'
				x_ch_repeat = 'x_fitted_' + re.sub(r'channel_\d','channel_4', ch_repeat)
				p1 = case.index_of_peaks_to_annotate[ch]
				p2 = case.index_of_peaks_to_annotate[ch_repeat]
				peaks1 = set()
				peaks2 = set()
				for i in p1:
					i_re = case.df[x_ch][i]
					for j in p2:
						j_re = case.df[x_ch_repeat][j]
						if abs(i_re-j_re) < 1.0:
							peaks1.add(i)
							peaks2.add(j)
				case.index_of_replicate_peaks[ch] = sorted(list(peaks1))
				case.index_of_replicate_peaks[ch_repeat] = sorted(list(peaks2))
	return case

def main():
	owd = os.getcwd()	# original working directory
	path = os.path.abspath(sys.argv[1])
	os.chdir(path)
	cases = organize_clonality_files(path)
	# output_path = os.path.join(path, '/plots')
	# if not os.path.exists(output_path): os.mkdir(output_path)
	for case_name in sorted(cases.keys()):
		case = cases[case_name]
		print('Processing {}'.format(case_name))
		case = gather_case_data(case, case_name, path)
		case = size_standard(case, ch_ss_num=4)
		case = find_artifactual_peaks(case)
		# case = baseline_correction_simple(case)
		case = baseline_correction_advanced(case, ch_list=channels_of_interest.keys(), distance=10)
		# case = pick_peak_one(case)
		# case = make_decay_curve(case)
		case = local_southern(case)
		case = index_of_peaks_to_annotate(case)
		case = replicate_peaks(case)
		plot_clonality_case(case, replicate_only=False, w=1050, h=350)

if __name__ == '__main__':
	main()