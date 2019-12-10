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
from scipy.signal import find_peaks, peak_prominences
from types import SimpleNamespace
from pprint import pprint
from itertools import combinations

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
	return cases

class Case(object):
	pass

def gather_case_data(cases, path):
	for case_name, case in cases.items():
		df = pd.DataFrame()
		for t, files in case.files.items():
			for f in files:
				df_t = pd.read_csv(os.path.join(path,f))
#				print(df_t.shape)
				df_t.columns = [pretty_name(c,t) for c in df_t.columns]
				df = pd.concat([df, df_t], axis=1, sort=False)
		df.name = case_name
		case.df = df
	return cases

def local_southern(cases):
	for case in cases.values():
		# print('local_southern now working on {}'.format(case.name))
		p_one = np.polyfit([x for x in case.ladder_x[0:3]], [100,200,300], 2)
		p_two = np.polyfit([x for x in case.ladder_x[1:4]], [200,300,400], 2)

		x_0_to_200 = [x*x*p_one[0] + x*p_one[1] + p_one[2] for x in case.df.index.tolist()]
		x_300_to_400 = [x*x*p_two[0] + x*p_two[1] + p_two[2] for x in case.df.index.tolist()]
		x_200_to_300 = [(x1+x2)/2 for x1,x2 in zip(*[x_0_to_200, x_300_to_400])]

		x_fitted = x_0_to_200[:case.ladder_x[1]] + x_200_to_300[case.ladder_x[1]:case.ladder_x[2]] + x_300_to_400[case.ladder_x[2]:]

		x_df = pd.DataFrame(x_fitted)
		x_df.columns = ['x_fitted']
		case.df = pd.concat([case.df, x_df], axis=1, sort=False)
	return cases

def plot_cases(cases):
	all_channels = {
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
				'SCL_channel_1':'blue'
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
				'TCRG-B_channel_2':[(80,110),(160,195)]
		}
	for case in cases.values():
		# print(case.df.columns)
		multipage = case.name + '.pdf'
		with PdfPages(multipage) as pdf:
			channels = {k:v for k,v in all_channels.items() if k in case.df.columns}
			for ch in channels.keys():
				png_name = case.name + '_' + ch + '.png'
				# print(png_name)
				ch_repeat = '_'.join([ch, 'repeat'])
				if ch_repeat in case.df.columns: num_rows = 2
				else: num_rows = 1
				p, axs = plt.subplots(nrows=num_rows, ncols=1)
				p.subplots_adjust(hspace=0.5)
				p.suptitle(case.name)
				if 'SCL' in ch:
					if case.ladder_success: c = 'green'
					else: c = 'red'
					for x in case.ladder_x:
						axs.plot(x,case.df[ch][x], 'o', fillstyle='none', color=c)
					axs.plot(case.df.index.tolist(), case.df[ch], linewidth=0.25, color=channels[ch])
					axs.plot(case.df.index.tolist(), case.df['decay'], linewidth=0.25, color=c)
				elif num_rows==2:
					axs[0].plot(case.df['x_fitted'], case.df[ch], linewidth=0.25, color=channels[ch])
					axs[1].plot(case.df['x_fitted'], case.df[ch_repeat], linewidth=0.25, color=channels[ch])
					axs[0].set_title(ch, fontdict={'fontsize': 8, 'fontweight': 'medium'})
					axs[1].set_title(ch_repeat, fontdict={'fontsize': 8, 'fontweight': 'medium'})
					for ax in axs:
						ax.set_xlim([75, 450])
						ax.set_ylabel('RFU', fontsize=6)
						ax.set_xlabel('Fragment Size', fontsize=6)
						ax.yaxis.set_tick_params(labelsize=6)
						for x_start,x_end in regions_of_interest[ch]:
							ax.axvspan(x_start, x_end, facecolor='black', alpha=0.05)
						autoscale_y(ax)
				elif num_rows==1:
					axs.plot(case.df['x_fitted'], case.df[ch], linewidth=0.25, color=channels[ch])
					axs.set_title(ch, fontdict={'fontsize': 8, 'fontweight': 'medium'})
					axs.set_xlim([75, 450])
					axs.set_ylabel('RFU', fontsize=6)
					axs.set_xlabel('Fragment Size', fontsize=6)
					axs.yaxis.set_tick_params(labelsize=6)
					if ch in regions_of_interest.keys():
						for x_start,x_end in regions_of_interest[ch]:
							axs.axvspan(x_start, x_end, facecolor='black', alpha=0.05)
					autoscale_y(axs)

				# plt.savefig(png_name, dpi=300)
				# plt.show()
				pdf.savefig()
				plt.close(p)
			print('Done making {}'.format(multipage))

def pick_peak_one(cases):
	for case in cases.values():
		case.ladder_success = False
		scldf = case.df['SCL_channel_1']
		print('Loading {}'.format(case.name))
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
			# print(case.name)
			# print(mask.size)
			# print(scldf.size)
			print('\tSkipping {} due to size mismatch, likely due to multiple files being added to the same column in the case DataFrame column'.format(case.name))
			for f in case.files['SCL']:
				print('\t\t{}'.format(f))
	return cases

def make_decay_curve(cases):
	for case in cases.values():
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
			case = evaluate_ladder(case, decay)
			if case.residual <= 10:
				case.ladder_success = True
				break
		case.decay_value = i
		# print('{}\ti = {}, residual = {}'.format(case.name, i, case.residual))
	return cases

def evaluate_ladder(case, decay):
	qualifying_peaks = [(x,y) for x,y in case.peaks if y > decay[x]]
	combos = [list(c) for c in combinations(qualifying_peaks, 3)]
	combos.sort(key=lambda coor:coor[0])
	case.ladder_x = [400,100,300,200]	# just some made up ladder
	case.residual = 1000000
	for combo in combos:
		ladder_x = [case.peak_one[0]] + [x for x,y in combo]
		poly_current, res_current, rank, singular_values, rcond = np.polyfit(ladder_x, [100,200,300,400], 1, full=True)
		res_current = res_current[0]
		if res_current < case.residual:
			case.residual = res_current
			case.ladder_x = ladder_x
	return case

def replace_height_with_prominence(cases):
	for case in cases.values():
		for col in case.df.columns:
			peaks_x, p = find_peaks(case.df[col], prominence=1)
			case.df[col].loc[peaks_x] = p['prominences']
	return cases

def build_ladder(choices, k):
	rox500_full = [35, 50, 75, 100, 139, 150, 160, 200, 250, 300, 340, 350, 400, 450, 490, 500]
	rox500_15 = [50, 75, 100, 139, 150, 160, 200, 250, 300, 340, 350, 400, 450, 490, 500]
	rox500 = np.array(rox500_15[0:k])
	if len(choices) <= k:
		print('\tWARNING: len(choices) = {}, k = {}'.format(len(choices), k))
	combos = np.array([c for c in combinations(choices, k)])
	pfit = np.polyfit(rox500, combos.T, deg=1, full=True)
	# print('\tsmallest residuals = {}'.format(np.amin(pfit[1])))
	i = np.argmin(pfit[1])
	# print('\tindex of smallest residuals = {}'.format(i))
	# print('\tx-coordinates of ladder with smallest residuals = {}'.format(combos[i]))
	best_ladder = combos[i]
	# print(residuals)
	print('\tchoices = {}, k = {}, combos = {}'.format(len(choices), k, len(combos)))
	return best_ladder
	# print(type(combos), combos.dtype)
	# print(combos)
	# print('\tlen(combos) = {}'.format(len(combos)))

def plot_channel_4(cases):
	k = 13
	last_peaks = []
	last_xy = []
	p, axs = plt.subplots(nrows=1, ncols=1)
	for case in cases.values():
		# if case.name == '19KD-323M0083':
		# rox500_list = [ch for ch in case.df.columns if 'channel_4' in ch and 'SCL' not in ch]
		rox500_list = [ch for ch in case.df.columns if 'channel_4' in ch]
		# p, axs = plt.subplots(nrows=1, ncols=1)
		for rox500 in rox500_list:
			label_name = '_'.join([case.name, rox500])
#			if case.df[rox500][case.df.index[-1]] > 50:
			# if rox500 == 'TCRB-C_channel_4':
			# p, axs = plt.subplots(nrows=1, ncols=1)
			peaks_x, _ = find_peaks(case.df[rox500], height=[20,1000], distance=30)
			if len(peaks_x) < k:
				print('case.name = {}, channel = {}'.format(case.name, rox500))
				print('WARNING: len(peaks_x) = {}, k = {}'.format(len(peaks_x), k))
				# plt.close(p)
				# p, axs = plt.subplots(nrows=1, ncols=1)
				axs.plot(case.df.index.tolist(), case.df[rox500], linewidth=0.25, label=label_name)
				axs.plot(peaks_x,case.df[rox500][peaks_x], 'o', fillstyle='none')
				# plt.show()
			else:
				ladder = build_ladder(peaks_x, k)
				ladder_y = [case.df[rox500][x] for x in ladder]
				if min(ladder_y) <= 50:
					axs.plot(peaks_x,case.df[rox500][peaks_x], 'o', fillstyle='none')
					axs.plot(ladder,case.df[rox500][ladder], 'x', fillstyle='none')
					axs.plot(case.df.index.tolist(), case.df[rox500], linewidth=0.25, label=label_name)
					plt.legend(prop={'size': 6, 'weight': 'medium'})
					last_peaks.append((label_name, peaks_x[-1]))
					last_xy.append((label_name, case.df[rox500][case.df.index[-1]]))
	plt.show()
	plt.close(p)
	last_peaks.sort(key=lambda e: e[1])
	last_xy.sort(key=lambda e: e[1])
	# for i in last_xy:
	# 	print(i)

def main():
	owd = os.getcwd()	# original working directory
	path = os.path.abspath(sys.argv[1])
	os.chdir(path)
	cases = organize_files(path)
	cases = gather_case_data(cases, path)
	# cases = replace_height_with_prominence(cases)
	plot_channel_4(cases)
	# cases = pick_peak_one(cases)
	# cases = make_decay_curve(cases)
	# cases = local_southern(cases)

	# if not os.path.exists(path + '/plots'):
	# 	os.mkdir(path +'/plots')
	# os.chdir(path + '/plots')

	# plot_cases(cases)

if __name__ == '__main__':
	main()