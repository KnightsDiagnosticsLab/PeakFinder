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
	channel = re.findall(r'channel_\d$', c)[0]
	if 'repeat' in c:
		pc = '_'.join([t, channel, 'repeat'])
	else:
		pc = '_'.join([t, channel])
	return pc

def organize_files(path):
	tests = [
				'IGH-A', 'IGH-B', 'IGH-C', 'IGK-A', 'IGK-B',
				'TCRB-A', 'TCRB-B', 'TCRB-C', 'TCRG-A', 'TCRG-B',
				'SCL'
			]

	# construct case list
	csv_list = [f for f in os.listdir(path) if f.endswith('.fsa')]
	case_names_as_llt = [re.findall(r'(\d\dKD-\d\d\dM\d\d\d\d)(-R)*', x) for x in csv_list]	# 'llt' is 'list of lists of tuple'
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
				df_t = pd.read_csv('/'.join([path,f]))
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
	channels = {
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
			for ch in channels.keys():
				ch_repeat = '_'.join([ch, 'repeat'])
				png_name = case.name + '_' + ch + '.png'
				plt.clf()
				if 'SCL' in ch:
					if case.ladder_success: c = 'green'
					else: c = 'red'
					print(png_name)
					p, ax = plt.subplots()
					for x in case.ladder_x:
						ax.plot(x,case.df[ch][x], 'o', fillstyle='none', color=c)
					ax.plot(case.df.index.tolist(), case.df[ch], linewidth=0.25, color=channels[ch])
					ax.plot(case.df.index.tolist(), case.df['decay'], linewidth=0.25, color=c)
					plt.savefig(png_name, dpi=300)
					pdf.savefig()
					plt.close(p)
				elif ch in case.df.columns and ch_repeat in case.df.columns:
					print(png_name)
					p, axs = plt.subplots(nrows=2, ncols=1)
					p.subplots_adjust(hspace=0.5)
					p.suptitle(case.name)

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

					plt.savefig(png_name, dpi=300)
					pdf.savefig()
					plt.close(p)

def pick_peak_one(cases):
	for case in cases.values():
		case.ladder_success = False
		scldf = case.df['SCL_channel_1']
		print('Now working on {}'.format(case.name))
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

def main():
	owd = os.getcwd()	# original working directory
	path = os.path.abspath(sys.argv[1])
	os.chdir(path)
	cases = organize_files(path)
	cases = gather_case_data(cases, path)
	# cases = replace_height_with_prominence(cases)
	cases = pick_peak_one(cases)
	cases = make_decay_curve(cases)
	cases = local_southern(cases)

	if not os.path.exists(owd + '/plots'):
		os.mkdir(owd +'/plots')
	os.chdir(owd + '/plots')

	plot_cases(cases)

if __name__ == '__main__':
	main()