#!/usr/bin/env python3

import re
import os
import sys
import pandas as pd
import easygui
import openpyxl
from openpyxl.styles import Border, Side, PatternFill, Font, GradientFill, Alignment
import string
import win32com.client as win32
import csv
from fsa import use_csv_module, fix_formatting

pd.set_option('display.max_columns', 20)

def build_results_dict(df=None):
	peaks = {}
	if not isinstance(df, pd.DataFrame):
		filename = easygui.fileopenbox(
			msg='Select results file')
		if filename is None:
			exit()
		df = use_csv_module(filename)
	df = df[['Sample File Name', 'Marker', 'Allele', 'Area']]
	# get rid of peaks that aren't assigned an allele
	df.dropna(axis=0, how='any', inplace=True)
	df = df[df['Allele'] != 'OL']		# get rid of 'off ladder' peaks
	df.reset_index(drop=True, inplace=True)
	# keys = []
	fnames = set()
	for i in df.index:
		# file_name = str(df.iloc[i]['Sample File Name']).upper()
		file_name = str(df.iloc[i]['Sample File Name'])
		fnames.add(file_name)
		# locus = str(df.iloc[i]['Marker']).upper()
		# allele = str(df.iloc[i]['Allele']).upper()
		locus = str(df.iloc[i]['Marker'])
		allele = str(df.iloc[i]['Allele'])
		key = (file_name, locus, allele)
		peaks[key] = peaks.get(key, 0) + int(df.iloc[i]['Area'])
	# for key in keys:
	# 	print(key)
	# assert len(keys) == len(set(keys))
	# for k,v in results.items():
	# 	print(k,v)
	# print(filename)
	# for x in fnames:
	#	 print('\t' + x)
	return peaks


def get_col_to_drop(df):
	col_to_drop = []
	for col in df.columns:
		if 'Unnamed' in col and df[col].isnull().all():
			col_to_drop.append(col)
	return col_to_drop


def build_profile_2(res, sample_name, template_path):
	cases = sorted(list({k[0] for k in res.keys()}))

	# owd = os.getcwd()  # original working directory
	# os.chdir(
	# 	r'X:\Hospital\Genetics Lab\DNA_Lab\3-Oncology Tests\Engraftment\Allele Charts')
	# msg = 'open template for ' + ', '.join(choices)
	# template = easygui.fileopenbox(title=msg)
	twd = os.path.dirname(os.path.abspath(template_path))
	# os.chdir(owd)
	# if template is None:
	# 	exit()

	# for choice in choices:
		# file_name = str(choice).upper()
	file_name = str(choice)
	df = pd.read_excel(template)
	# temp_filename = 'temp1_' + file_name.replace('.FSA', '.xlsx')
	# df.to_excel(temp_filename, index=False)

	replacement_dict = {
		'THO1': 'TH01',
		'Amelogenin': 'AMEL',
		'amelogenin': 'AMEL',
		'AMELOGENIN': 'AMEL',
		'Recipient (Host) Alleles': 'Host',
		'VWA': 'vWA',
		'vwa': 'vWA',
		'vWa': 'vWA',
		'VWa': 'vWA',
		'VwA': 'vWA',
	}
	df.replace(to_replace=replacement_dict, inplace=True)
	col_to_drop = get_col_to_drop(df)
	df.drop(axis=1, columns=col_to_drop, inplace=True)
	# temp_filename = 'temp2_' + file_name.replace('.FSA', '.xlsx')
	# df.to_excel(temp_filename, index=False)

	# get locations of 'Allele'
	allele_ij = []
	for i in df.index:
		for j, v in enumerate(df.iloc[i]):
			if v == 'Allele':
				allele_ij.append([i, j])

	for i, j in allele_ij:
		# locus = str(df.iloc[i, 0]).upper()
		locus = str(df.iloc[i, 0])
		for k in range(1, j):
			# x = str(df.iloc[i, j + k]).upper()
			x = str(df.iloc[i, j + k])
			if len(x) > 0 and x != pd.np.nan and x != 'NAN':
				key = (file_name, locus, x)
				df.iat[i + 1, j + k] = res.get(key, 0)
			else:
				df.iat[i + 1, j + k] = pd.np.nan

	# Get rid of the remaining 'Unnamed: #' column labels
	df.rename(columns=lambda x: re.sub(r'Unnamed.*', '', x), inplace=True)

	# Write the output
	case_name = re.findall(r'(\d\dKD.*)_PTE', file_name)[0]
	print('case_name = {}'.format(case_name))
	output_file_name = re.sub(r'_PTE.*$', '.xlsx', file_name)
	# output_file_name = file_name.replace('.FSA', '.xlsx')
	header = get_header(template)
	patient_name = header.center.text
	output_file_name = ' '.join([patient_name, output_file_name])
	os.chdir(twd)
	output_file_name = easygui.filesavebox(
		msg='Save As', default=output_file_name)
	if not output_file_name.endswith('.xlsx'):
		output_file_name = output_file_name + '.xlsx'
	df.to_excel(output_file_name, index=False)
	# insert_formulae(output_file_name, template)
	# insert_formulae_2(output_file_name)
	fix_formatting(output_file_name, header, case_name)
	# insert_header(output_file_name, template)




def build_profile_1(res):
	cases = sorted(list({k[0] for k in res.keys()}))
	choices = easygui.multchoicebox(
		msg='Pick cases that share a template',
		choices=cases,
		preselect=None)
	# print('choices = {}'.format(choices))
	if choices is None:
		exit()

	owd = os.getcwd()  # original working directory
	os.chdir(
		r'X:\Hospital\Genetics Lab\DNA_Lab\3-Oncology Tests\Engraftment\Allele Charts')
	msg = 'open template for ' + ', '.join(choices)
	template = easygui.fileopenbox(title=msg)
	twd = os.path.dirname(os.path.abspath(template))
	# os.chdir(owd)
	if template is None:
		exit()

	for choice in choices:
		# file_name = str(choice).upper()
		file_name = str(choice)
		df = pd.read_excel(template)
		# temp_filename = 'temp1_' + file_name.replace('.FSA', '.xlsx')
		# df.to_excel(temp_filename, index=False)

		replacement_dict = {
			'THO1': 'TH01',
			'Amelogenin': 'AMEL',
			'amelogenin': 'AMEL',
			'AMELOGENIN': 'AMEL',
			'Recipient (Host) Alleles': 'Host',
			'VWA': 'vWA',
			'vwa': 'vWA',
			'vWa': 'vWA',
			'VWa': 'vWA',
			'VwA': 'vWA',
		}
		df.replace(to_replace=replacement_dict, inplace=True)
		col_to_drop = get_col_to_drop(df)
		df.drop(axis=1, columns=col_to_drop, inplace=True)
		# temp_filename = 'temp2_' + file_name.replace('.FSA', '.xlsx')
		# df.to_excel(temp_filename, index=False)

		# get locations of 'Allele'
		allele_ij = []
		for i in df.index:
			for j, v in enumerate(df.iloc[i]):
				if v == 'Allele':
					allele_ij.append([i, j])

		for i, j in allele_ij:
			# locus = str(df.iloc[i, 0]).upper()
			locus = str(df.iloc[i, 0])
			for k in range(1, j):
				# x = str(df.iloc[i, j + k]).upper()
				x = str(df.iloc[i, j + k])
				if len(x) > 0 and x != pd.np.nan and x != 'NAN':
					key = (file_name, locus, x)
					df.iat[i + 1, j + k] = res.get(key, 0)
				else:
					df.iat[i + 1, j + k] = pd.np.nan

		# Get rid of the remaining 'Unnamed: #' column labels
		df.rename(columns=lambda x: re.sub(r'Unnamed.*', '', x), inplace=True)

		# Write the output
		case_name = re.findall(r'(\d\dKD.*)_PTE', file_name)[0]
		print('case_name = {}'.format(case_name))
		output_file_name = re.sub(r'_PTE.*$', '.xlsx', file_name)
		# output_file_name = file_name.replace('.FSA', '.xlsx')
		header = get_header(template)
		patient_name = header.center.text
		output_file_name = ' '.join([patient_name, output_file_name])
		os.chdir(twd)
		output_file_name = easygui.filesavebox(
			msg='Save As', default=output_file_name)
		if not output_file_name.endswith('.xlsx'):
			output_file_name = output_file_name + '.xlsx'
		df.to_excel(output_file_name, index=False)
		# insert_formulae(output_file_name, template)
		# insert_formulae_2(output_file_name)
		fix_formatting(output_file_name, header, case_name)
		# insert_header(output_file_name, template)


def get_header(template):
	if template.endswith('.xlsx'):
		pass
	elif template.endswith('.xls') and os.path.isfile(template + 'x'):
		template = template + 'x'
	else:
		excel = win32.gencache.EnsureDispatch('Excel.Application')
		wb = excel.Workbooks.Open(template)

		template = template + 'x'
		# FileFormat = 51 is for .xlsx extension
		wb.SaveAs(template, FileFormat=51)
		wb.Close()  # FileFormat = 56 is for .xls extension
		excel.Application.Quit()
	assert template.endswith('.xlsx')

	wb = openpyxl.load_workbook(template)
	ws = wb.worksheets[0]
	header = ws.oddHeader
	return header


def location_of_value(ws, val):
	loc = None
	for j in range(1, ws.max_row + 1):
		for i in range(0, ws.max_column):
			c = string.ascii_uppercase[i]
			cell = ws[c + str(j)]
			if val == cell.value:
				loc = (c, j)
				print('loc of {} = {}'.format(val, loc))
				return loc


def insert_formulae_2(filename):
	wb = openpyxl.load_workbook(filename)
	ws = wb.worksheets[0]

	allele_loc = location_of_value(ws, 'Allele')
	return None


def insert_formulae(filename, template):
	if template.endswith('.xls'):
		excel = win32.gencache.EnsureDispatch('Excel.Application')
		wb = excel.Workbooks.Open(template)

		template = template + 'x'
		# FileFormat = 51 is for .xlsx extension
		wb.SaveAs(template, FileFormat=51)
		wb.Close()  # FileFormat = 56 is for .xls extension
		excel.Application.Quit()
	assert template.endswith('.xlsx')
	wbt = openpyxl.load_workbook(template)
	wst = wbt.worksheets[0]

	# find cell with '% Host'
	host_loc_t = location_of_value(wst, '% Host')
	formula_dict = {}

	for i in range(host_loc_t[1] + 1, wst.max_row + 1):
		cell1 = wst['A' + str(i)]
		cell2 = wst[host_loc_t[0] + str(i)]
		print(cell1.value, cell2.value)
		formula_dict[cell1.value] = cell2.value
	formula_dict.pop(None)

	''' open outputfile and insert formulae
	'''
	wb = openpyxl.load_workbook(filename)
	ws = wb.worksheets[0]

	host_loc = location_of_value(ws, '% Host')

	for i in range(host_loc[1] + 1, wst.max_row + 1):
		cell1 = ws['A' + str(i)]
		cell2 = ws[host_loc[0] + str(i)]
		print(cell1.value, cell2.value)
		cell2.value = formula_dict.get(cell1.value, cell2.value)
		print(cell1.value, cell2.value)
		print('')
	wb.save(filename)







def main():
	owd = os.getcwd()  # original working directory
	os.chdir(r'X:\Hospital\Genetics Lab\DNA_Lab\Ghani')
	results = build_results_dict()
	f = [k[0] for k in results.keys()]
	f = set(f)
	for x in f:
		print(x)

	while True:
		build_profile(results)


if __name__ == '__main__':
	main()
