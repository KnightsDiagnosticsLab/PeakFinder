#!/usr/bin/env python3

# Importing Packages
import os
import sys
import re
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps
from itertools import combinations
from outliers import smirnov_grubbs as grubbs

from bokeh.io import output_file, show, save
from bokeh.layouts import column
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import BoxAnnotation, Label, Range1d, WheelZoomTool, ResetTool, PanTool, LegendItem, Legend, LabelSet
from bokeh.core.validation.warnings import FIXED_SIZING_MODE
from bokeh.core.validation import silence
from bokeh.models.markers import Diamond

import clonality as clo

TOOLTIPS = [("(x,y)", "($x{1.1}, $y{int})")]
silence(FIXED_SIZING_MODE, True)

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
	return cases

def highlight_roi_PTE(case, ch, p):
	ch_num = re.findall(r'channel_\d$', ch)[0]
	legends = []
	for allele_name, allele in alleles[ch_num].items():
		x_left = allele['range'][0]
		x_right = allele['range'][1]
		roi_color = allele['color']
		dummy_dot = p.line([0,0],[1,1], line_width=20, color=roi_color, alpha=0.10)
		roi = BoxAnnotation(left=x_left, right=x_right, fill_color=roi_color, fill_alpha=0.05)
		p.add_layout(roi)
		legends.append(LegendItem(label=allele_name, renderers=[dummy_dot]))
	p.add_layout(Legend(items=legends, location='top_right'))
		# print(p.legend.items)
	return p

def plot_PTE_case(case, plot_dict, w=1050, h=200, ch_ss_num=5):
	ch_ss = 'channel_' + str(ch_ss_num)
	# ch_list = [ch for ch in case.df.columns if 'x_fitted' not in ch and ch not in plot_dict.keys()]
	ch_list = [ch for ch in case.df.columns if 'x_fitted' not in ch]
	for ch in ch_list:
		ch_num = re.findall(r'channel_\d$', ch)[0]
		x_col_name = 'x_fitted_' + re.sub(r'channel_\d',ch_ss, ch)
		p = figure(tools='pan,wheel_zoom,reset',title=ch, width=w, height=h, x_axis_label='fragment size', y_axis_label='RFU', tooltips=TOOLTIPS)
		p.toolbar.logo = None
		if ch_ss not in ch:
			p.x_range = Range1d(75,400)
			x = case.df[x_col_name].to_list()
			y = case.df[ch].to_list()
			p.line(x, y, line_width=0.5, color=clo.channel_colors.get(ch_num, 'blue'))
			# print(case.plot_labels.get(ch,[]))
			# print(ch)
			source = pd.DataFrame.from_records(case.plot_labels.get(ch,[]), columns=['x','y','label','left_bases','right_bases'])
			labels = LabelSet(
				angle=1,
				x='x',
				y='y',
				text='label',
				x_offset=0,
				y_offset=2,
				source=ColumnDataSource(source),
				render_mode='canvas',
				text_font_size='10pt'
			)
			p.add_layout(labels)
			p = highlight_roi_PTE(case, ch, p)
			p.x(x='x',y='y', source=source)
			for i in case.abberant_peaks.get(ch,[]):
				p.circle(x=case.df[x_col_name][i], y=case.df[ch][i], color='red', size=10, alpha=0.25)
			# if 'Allelic' not in ch:
			# 	for y, x1, x2 in case.widths[ch]:
			# 		p.line([case.df[x_col_name][x1],y],[case.df[x_col_name][x2],y])
			plot_dict[ch] = p

			# p.x(x='left_bases',y=0, source=source, color='red')
			# p.diamond(x='right_bases',y=0, source=source, color='green')
			plot_dict[ch] = p
		else:
			plot_dict = clo.plot_size_standard(case, ch, plot_dict, w, h=400, ch_ss_num=5)
	# print('len(plot_dict.values()) = {}'.format(len(plot_dict.values())))
	# print(type(plot_dict.values()))
	return plot_dict



def find_peaks_in_range(case, ch, x_col_name, start, end, h=100, d=10):
	peaks, _ = find_peaks(case.df[ch], height=h, distance=d)
	peaks = [i for i in peaks if case.df[x_col_name][i] >= start and case.df[x_col_name][i] <= end]
	# print('\tlen(peaks) = {}'.format(len(peaks)))
	return peaks

def build_allelic_ladder(case, allelic_case_names, ch_ss_num=5):
	if 'Allelic' in case.name:
		allelic_case_names.append(case.name)
		ch_ss = 'channel_' + str(ch_ss_num)
		channels = [ch for ch in case.df.columns if 'x_fitted' not in ch and ch_ss not in ch]
		x_col_name = [ch for ch in case.df.columns if 'x_fitted' in ch][0]
		for ch in channels:
			ch_num = re.findall(r'channel_\d$', ch)[0]
			# p = figure(tools='pan,wheel_zoom,reset',title=ch, width=1050, height=400, x_axis_label='fragment size', y_axis_label='RFU', tooltips=TOOLTIPS)
			# p.x_range = Range1d(75,400)
			x = case.df[x_col_name].to_list()
			y = case.df[ch].to_list()
			# p.line(x, y, line_width=0.5, color=clo.channel_colors.get(ch_num, 'blue'))
			# print(ch)
			for region in alleles[ch_num].keys():
				start, end = alleles[ch_num][region]['range']
				stock_ladder = alleles[ch_num][region]['stock ladder']
				allelic_labels = alleles[ch_num][region]['allelic labels']
				choices = find_peaks_in_range(case, ch, x_col_name, start, end, h=120, d=5)
				if len(choices) < len(stock_ladder):
					print('\tWARNING: len(choices) = {}, k = {}'.format(len(choices), len(stock_ladder)))
					# print('\tlen(stock_ladder = {})'.format(len(stock_ladder)))
					# print('\tlen(choices) = {}'.format(len(choices)))
				else:
					X = np.array([sorted(list(c)) for c in combinations(choices, len(stock_ladder))])

					pfit_zx = np.polyfit(stock_ladder, X.T, deg=1, full=True)
					residuals_zx = pfit_zx[1]
					X_mean = np.expand_dims(np.mean(X,axis=1),axis=1)
					R_sq_zx = 1.0 - (np.square(residuals_zx) / np.sum(np.square(X - X_mean)))
					if len(R_sq_zx) > 0:	# it's zero when a line is being fit to only two points.
						ranked_R_sq, indices = np.unique(R_sq_zx, return_index=True)
						indices = indices.tolist()
						indices.reverse()
						ladder_i = X[indices[0]]
						ladder = case.df[x_col_name][ladder_i].to_list()
						alleles[ch_num][region][case.name] = {'ladder_i':ladder_i, 'ladder':ladder}
					else:
						ladder_i = X[0]
						ladder = case.df[x_col_name][ladder_i].to_list()
				x_peaks = case.df[x_col_name][ladder_i]
				y_peaks = case.df[ch][ladder_i]
				# p.x(x_peaks, y_peaks)
				# for x,y,label in zip(x_peaks, y_peaks, allelic_labels):
				# 	mytext = Label(angle=1, x=x, y=int(y), text=str(label), x_offset=0, y_offset=2, text_font_size='8pt')
				# 	p.add_layout(mytext)
				plot_labels = case.plot_labels.get(ch,[])
				# print('plot_labels = {}'.format(len(plot_labels)))
				plot_labels.extend([(x,y,label,None,None) for x,y,label in zip(x_peaks,y_peaks,allelic_labels)])
				# print('plot_labels_new = {}'.format(len(plot_labels)))
				case.plot_labels[ch] = plot_labels
		# plots = column(plot_list)
		# show(plots)
	return case, allelic_case_names

def label_alleles(case, allelic_case, w=900, h=400, ch_ss_num=5):
	# print(case.name)
	pte_run = re.findall(r'PTE-\d+-\d+', case.name)[0]
	ch_ss = 'channel_' + str(ch_ss_num)
	ch_list = [ch for ch in case.df.columns if 'x_fitted' not in ch and ch_ss not in ch]

	for ch in ch_list:
		plot_label = []
		x_col_name = 'x_fitted_' + re.sub(r'channel_\d',ch_ss, ch)
		ch_num = re.findall(r'channel_\d$', ch)[0]
		ch_allelic = [ch_a for ch_a in allelic_case.df.columns if ch_num in ch_a][0]
		peaks_i, props = find_peaks(case.df[ch], height=100, prominence=100, distance=10)
		left_bases = props['left_bases']
		right_bases = props['right_bases']
		peaks_x = case.df[x_col_name][peaks_i].to_list()
		peaks_y = case.df[ch][peaks_i].to_list()
		allelic_ladder = allelic_case.plot_labels[ch_allelic]
		
		# broadcast subtraction
		X = np.matrix(peaks_x).T
		XI = np.matrix(peaks_i).T
		Y = np.matrix(peaks_y).T
		# print('X = {}'.format(X))
		A = np.matrix([a for a,*_ in allelic_ladder])
		# print('A = {}'.format(A))
		R = np.absolute(X-A)
		# print('R = {}'.format(R))
		IJ = np.nonzero(R<1.2)
		I = IJ[0]
		J = IJ[1]
		# print('I = {}'.format(I))
		X = np.squeeze(np.asarray(X[I]))
		XI = np.squeeze(np.asarray(XI[I]))
		# print('X = {}'.format(X))
		# print('XI = {}'.format(XI))
		Y = np.squeeze(np.asarray(Y[I]))
		# print('J = {}'.format(J))
		labels = [allelic_ladder[j][2] for j in J]
		left_bases = left_bases[I]
		right_bases = right_bases[I]
		# compute area of each peak
		# labels = []
		# print(left_bases)
		# for lb, rb, l in zip(left_bases, right_bases, labels_temp):
		# 	x = case.df[x_col_name][lb:rb].to_list()
		# 	# print('x = {}'.format(x))
		# 	y = case.df[ch][lb:rb].to_list()
		# 	# print('y = {}'.format(y))
		# 	area = simps(y=y,x=x)
		# 	# print('area = {}'.format(area))
		# 	labels.append([l,int(area)])
		"""	Too many left & right bases. Need to use the matrix mask to reduce bases.
		"""
		plot_label = [(x,y,l,lb,rb) for x,y,l,lb,rb in zip(X,Y,labels,left_bases,right_bases)]
		plot_label_temp = {}
		for x,y,l,lb,rb in plot_label:
			t = plot_label_temp.get(x,[])
			t.append(l)
			plot_label_temp[x] = t

		""" second attempt using peak_widths
		"""
		# widths = peak_widths(x=case.df[ch],peaks=XI)
		# print(widths)
		# hlines = [(y,x1,x2) for y,x1,x2 in zip(widths[1], widths[2], widths[3])]
		# case.widths[ch] = hlines
		# results_full = peak_widths(case.df[ch], peaks, rel_height=1)
		# print('widths = {}'.format(widths))
		# print('ch = {}'.format(ch))
		# print('len(I) = {}'.format(len(I)))
		# print('len(widths) = {}'.format(len(widths)))
		# for w in widths:
		# 	pass
		
		plot_label = [(x,y,plot_label_temp[x],case.df[x_col_name][lb],case.df[x_col_name][rb]) for x,y,lb,rb in zip(X,Y,left_bases,right_bases)]

		case.plot_labels[ch] = plot_label


	return case

def main():
	owd = os.getcwd()	# original working directory
	path = os.path.abspath(sys.argv[1])
	os.chdir(path)
	cases = organize_PTE_files(path)
	plot_dict = {}
	allelic_case_names = []

	for case in cases.values():
		print('working on {}'.format(case.name))
		case = gather_PTE_case_data(case, path)
		# case = clo.baseline_correction_simple(case, ch_ss_num=5)
		case = clo.baseline_correction_advanced(case, ch_ss_num=5, prominence=10)
		# case = clo.baseline_correction_simple(case, ch_ss_num=5)
		# case = clo.baseline_correction_upside_down(case, ch_ss_num=5, iterations=5)
		case = clo.size_standard(case, ch_ss_num=5)
		case = clo.local_southern(case)
		case, allelic_case_names = build_allelic_ladder(case, allelic_case_names, ch_ss_num=5)

	allelic_case = cases[allelic_case_names[0]]
	non_allelic_case_names = [case_name for case_name,case in cases.items() if 'Allelic' not in case_name]
	for case_name in non_allelic_case_names:
		case = cases[case_name]
		case = label_alleles(case, allelic_case, ch_ss_num=5)

	for case in cases.values():
		plot_dict = plot_PTE_case(case, plot_dict, w=1050, h=500)

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

"""	Next steps
	1. Correctly label regions and peaks on the allelic ladder.
	2. Use allelic ladder to annotate sample peaks.
		a. May need to deal with stutter vs. true allelic peak
	3. Get area under each peak.
		a. Given a list of peaks, get their bases.
		b. Include stutter in area calculation or not? Won't matter so long as we are consistent. Easier to exclude stutter.
"""

alleles = {
	'channel_1': {
		'D8S1179': {
			'range':(115,175),
			'chromosome location':8,
			'allelic labels': (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
			'stock ladder': (122.49, 126.56, 130.66, 134.8, 138.98, 143.58, 148.03, 152.43, 156.73, 160.93, 165.03, 169.1),
			'dye label': '6-FAM',
			'control DNA 9947A':(13,13),
			'color':'blue'
		},
		'D21S11': {
			'range':(175,245),
			'chromosome location':'21q11.2-q21',
			'allelic labels': (24, 24.2, 25, 26, 27, 28, 28.2, 29, 29.2, 30, 30.2, 31, 31.2, 32, 32.2, 33, 33.2, 34, 34.2, 35, 35.2, 36, 37, 38),
			'stock ladder': (184.41, 186.39, 188.35, 192.27, 196.21, 200.06, 202.03, 204.02, 206.08, 208.06, 210.03, 212.04, 214.03, 216.04, 218.03, 220.05, 221.98, 224.12, 226.03, 228.1, 230.03, 232.02, 236.08, 240.04),
			'dye label': '6-FAM',
			'control DNA 9947A':(30,30),
			'color':'green'
		},
		'D7S820': {
			'range':(245,297),
			'chromosome location':'7q11.21-22',
			'allelic labels': (6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
			'stock ladder': (255.08, 259.13, 263.16, 267.19, 271.25, 275.28, 279.34, 283.38, 287.44, 291.51),
			'dye label': '6-FAM',
			'control DNA 9947A':(10,11),
			'color':'red'
		},
		'CSF1PO': {
			'range':(297,360),
			'chromosome location':'5q33.3-34',
			'allelic labels': (6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
			'stock ladder': (303.99, 308.04, 312.1, 316.13, 320.18, 324.24, 328.3, 332.36, 336.39, 340.42),
			'dye label': '6-FAM',
			'control DNA 9947A':(10,12),
			'color':'purple'
		},
	},
	'channel_2': {
		'D3S1358': {
			'range':(100,150),
			'chromosome location':'3p',
			'allelic labels': (12, 13, 14, 15, 16, 17, 18, 19),
			'stock ladder': (111.12, 115.23, 119.2, 123.14, 127.32, 131.54, 135.64, 139.72),
			'dye label': 'VIC',
			'control DNA 9947A':(14,15),
			'color':'blue'
		},
		'TH01': {
			'range':(155,210),
			'chromosome location':'11p15.5',
			'allelic labels': (4, 5, 6, 7, 8, 9, 9.3, 10, 11, 13.3),
			'stock ladder': (162.72, 166.78, 170.82, 174.83, 178.84, 182.82, 185.84, 186.77, 190.71, 201.48),
			'dye label': 'VIC',
			'control DNA 9947A':(8,9.3),
			'color':'green'
		},
		'D13S317': {
			'range':(210,260),
			'chromosome location':'13q22-31',
			'allelic labels': (8, 9, 10, 11, 12, 13, 14, 15),
			'stock ladder': (216.36, 220.34, 224.32, 228.31, 282.42, 236.3, 240.24, 244.23),
			'dye label': 'VIC',
			'control DNA 9947A':(11,11),
			'color':'red'
		},
		'D16S539': {
			'range':(240,300),
			'chromosome location':'16q24-qter',
			'allelic labels': (5, 8, 9, 10, 11, 12,13, 14, 15),
			'stock ladder': (252.01, 264, 268, 272, 276.02, 280.03, 284.05, 288.08, 292.12),
			'dye label': 'VIC',
			'control DNA 9947A':(11,12),
			'color':'purple'
		},
		'D2S1338': {
			'range':(300,375),
			'chromosome location':'2q35-37.1',
			'allelic labels': (15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28),
			'stock ladder': (306.27, 310.35, 314.39, 318.45, 322.52, 326.58, 330.66, 334.71, 338.74, 342.75, 346.78, 350.77, 354.69, 358.87),
			'dye label': 'VIC',
			'control DNA 9947A':(19,23),
			'color':'yellow'
		},
	},
	'channel_3': {
		'D19S433': {
			'range':(90,145),
			'chromosome location':'19q12-13.1',
			'allelic labels': (9, 10, 11, 12, 12.2, 13, 13.2, 14, 14.2, 15, 15.2, 16, 16.2, 17, 17.2),
			'stock ladder': (101.25, 105.16, 109.09, 113.04, 115.06, 117.02, 119.03, 121.02, 123.05, 125.03, 127.08, 129.08, 131.13, 133.16, 135.23),
			'dye label': 'NED',
			'control DNA 9947A':(14,15),
			'color':'blue'
		},
		'vWA': {
			'range':(145,215),
			'chromosome location':'12p12-pter',
			'allelic labels': (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),
			'stock ladder': (154.07, 158.26, 162.42, 166.66, 170.59, 174.62, 178.61, 182.54, 186.5, 190.43, 194.29, 198.17, 202.01, 206.36),
			'dye label': 'NED',
			'control DNA 9947A':(17,18),
			'color':'green'
		},
		'TPOX': {
			'range':(215,265),
			'chromosome location':'2p23-2per',
			'allelic labels': (6, 7, 8, 9, 10, 11, 12, 13),
			'stock ladder': (221.82, 225.8, 229.79, 233.77, 237.76, 241.75, 245.78, 249.76),
			'dye label': 'NED',
			'control DNA 9947A':(8,8),
			'color':'red'
		},
		'D18S51': {
			'range':(255,375),
			'chromosome location':'18q21.3',
			'allelic labels': (7, 9, 10, 10.2, 11, 12, 13, 13.2, 14, 14.2, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
			'stock ladder': (261.8, 269.94, 274.02, 276.03, 278.11, 282.2, 286.29, 288.29, 290.38, 292.39, 294.48, 298.57, 302.69, 306.83, 310.96, 315.08, 319.2, 323.39, 327.46, 331.59, 335.69, 339.8, 343.87),
			'dye label': 'NED',
			'control DNA 9947A':(15,19),
			'color':'purple'
		},
	},
	'channel_4': {
		'Amelogenin': {
			'range':(90,120),
			'chromosome location':'X: p22.1-22.3, Y: p11.2',
			'allelic labels': ('X', 'Y'),
			'stock ladder': (106.03 , 111.69),
			'dye label': 'PET',
			'control DNA 9947A':('X'),
			'color':'blue'
		},
		'D5S818': {
			'range':(120,180),
			'chromosome location':'5q21-31',
			'allelic labels': (7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
			'stock ladder': (133.69, 137.8, 142.17, 146.64, 151.05, 155.32, 159.55, 163.63, 167.68, 171.7),
			'dye label': 'PET',
			'control DNA 9947A':(11,11),
			'color':'green'
		},
		'FGA_low': {
			'range':(200,290),
			'chromosome location':'4q28',
			'allelic labels': (17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26.2, 27, 28, 29, 30, 30.2, 31.2, 32.2, 33.2),
			'stock ladder': (214.11, 218.14, 222.17, 226.21, 230.26, 234.29, 238.33, 242.37, 246.42, 250.48, 252.49, 254.5, 258.55, 262.63, 266.72, 268.53, 272.62, 276.71, 280.77),
			'dye label': 'PET',
			'control DNA 9947A':(23,24),
			'color':'red'
		},
		'FGA_high': {
			'range':(310,375),
			'chromosome location':'4q28',
			'allelic labels': (42.2, 43.2, 44.2, 45.2, 46.2, 47.2, 48.2, 50.2, 51.2),
			'stock ladder': (317.89, 322.01, 326.14, 330.28, 334.28, 338.37, 342.51, 350.59, 354.54),
			'dye label': 'PET',
			'control DNA 9947A':(23,24),
			'color':'purple'
		},
	}
}

if __name__ == '__main__':
	main()
