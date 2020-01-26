#!/usr/bin/env python3

import re
import os
import sys
import pandas as pd
import easygui
import openpyxl
from openpyxl.styles import Border, Side, PatternFill, Font, GradientFill, Alignment
from openpyxl.utils.cell import coordinate_from_string, column_index_from_string
import win32com.client as win32
import csv
from fsa import *
import random

from pprint import pprint

pd.set_option('display.max_columns', 20)

def build_results_dict(df=None):
	peaks = {}
	if df is not None:
		# print('\t\t\tThis is inside of build_results_dict')
		# print(df)
	# if not isinstance(df, pd.DataFrame):
	# 	filename = easygui.fileopenbox(
	# 		msg='Select results file')
	# 	if filename is None:
	# 		exit()
	# 	df = use_csv_module(filename)
		df = df[['Sample File Name', 'Marker', 'Allele', 'Area']]

		'''	Get rid of peaks that aren't assigned an allele '''
		df = df.dropna(axis=0, how='any', inplace=False)

		'''	Get rid of OL (off ladder) peaks '''
		df = df[df['Allele'] != 'OL']
		df = df.reset_index(drop=True, inplace=False)

		fnames = set()
		for i in df.index:
			file_name = str(df.iloc[i]['Sample File Name'])
			fnames.add(file_name)
			locus = str(df.iloc[i]['Marker'])
			allele = str(df.iloc[i]['Allele'])
			key = (file_name, locus, allele)
			peaks[key] = peaks.get(key, 0) + int(df.iloc[i]['Area'])
	# print(peaks)
	return peaks


def build_profile_x(template_path, sample_name='', res={}):
	df = pd.DataFrame()

	print('Now running build_profile_2')

	replacement_dict_no_regex = {
		'THO1': 'TH01',
		'Recipient (Host) Alleles': 'Host',
	}

	replacement_dict_yes_regex = {
		'vwa': 'vWA',
		'amelogenin': 'AMEL',
	}

	if os.path.isfile(str(template_path)):
		# print(template_path)
		wb = openpyxl.load_workbook(template_path)
		ws = wb.worksheets[0]

		ws = replace_cell_values(ws, replacement_dict_no_regex, regex=False)
		ws = replace_cell_values(ws, replacement_dict_yes_regex, regex=True)

		df = pd.DataFrame(ws.values)
		print(df)

		allele_locs = locations_of_value(ws, 'Allele')

		'''	replace pointer formulae with what they refer to '''
		for loc in allele_locs:
			coor = coordinate_from_string(loc)
			c = column_index_from_string(coor[0])
			r = coor[1] + 1		# formula will be one row down from 'Allele'
			cell = ws.cell(column=c, row=r)
			if str(cell.value).startswith('='):
				ref_coor = str(cell.value).replace('=', '')
				cell.value = ws[ref_coor].value


	
	return df


def build_profile_2(template, sample_name='', res={}):
	print('Now running build_profile_2')
	# pprint(res)
	df = pd.DataFrame()

	if os.path.isfile(str(template)):
		# print(template)
		wb = openpyxl.load_workbook(template)
		ws = wb.worksheets[0]

		'''	Insert case number near top '''
		case_name = re.sub(r'_PTE.*$', '', sample_name)
		loc = location_of_value(ws, 'Post-T:')
		if loc is not None:
			cell = ws[chr(ord(loc[0]) + 1) + str(loc[1])]
			cell.value = case_name
		
		df = pd.DataFrame(ws.values)
		wb.close()
	elif isinstance(template, pd.DataFrame):
		df = template
	else:
		return df

	replacement_dict_no_regex = {
		'THO1': 'TH01',
		'Recipient (Host) Alleles': 'Host',
	}
	replacement_dict_yes_regex = {
		'[vV][wW][aA]':'vWA',
		'[aA][mM][eE][lL][oO][gG][eE][nN][iI][nN]':'AMEL',
	}

	# df.replace(to_replace='[vV][wW][aA]', value='vWA', regex=True, inplace=True)
	# df.replace(to_replace='[aA][mM][eE][lL][oO][gG][lL][oO][bB][iI][nN]', value='AMEL', regex=True, inplace=True)
	df.replace(to_replace=replacement_dict_no_regex, inplace=True, regex=False)
	df.replace(to_replace=replacement_dict_yes_regex, inplace=True, regex=True)
	# print(df)

	''' Get locations of 'Allele' '''
	allele_ij = []
	for i in df.index:
		for j, v in enumerate(df.iloc[i]):
			if v == 'Allele':
				allele_ij.append([i, j])
	# print(allele_ij)

	'''	replace cells that reference other cells with the ref cell's value '''
	for i, j in allele_ij:
		locus = str(df.iloc[i, 0])
		for k in range(1, j):
			x = str(df.iloc[i, j + k])
			if x.startswith('='):
				coor_ws_text = x.replace('=','')
				coor_ws_num = coordinate_from_string(coor_ws_text)
				coor_idx = column_index_from_string(coor_ws_num[0]) + 1
				coor_col = coor_ws_num[1]+1
				df.iloc[i, j + k] = df.iloc[coor_idx, coor_col]

	'''	fix the formatting of cells that have a trailing zero '''
	for i, j in allele_ij:
		locus = str(df.iloc[i, 0])
		for k in range(1, j):
			x = str(df.iloc[i, j + k])
			if x.endswith('.0'):
				df.iloc[i, j + k] = x.replace('.0','')

	''' Insert area values '''
	# pprint(res)
	for i, j in allele_ij:
		locus = str(df.iloc[i, 0])
		for k in range(1, j):
			x = str(df.iloc[i, j + k])
			# if len(x) > 0 and x != pd.np.nan and x != 'NAN':
			key = (str(sample_name), locus, x)
			val = res.get(key, pd.np.nan)
			# print(key, val)
			df.iat[i + 1, j + k] = res.get(key, pd.np.nan)
			# print('\tcoor = {}, value = {}'.format(coor, x))
			# else:
			# 	df.iat[i + 1, j + k] = pd.np.nan


	'''	Drop empty columns. Note that this should only be done after formulae are replaced with values '''
	col_to_drop = get_col_to_drop(df)
	df.drop(axis=1, columns=col_to_drop, inplace=True)

	'''	Get rid of the remaining 'Unnamed: #' column labels '''
	df.rename(columns=lambda x: re.sub(r'Unnamed.*', '', str(x)), inplace=True)

	'''	TO DO
		fix circular reference formulae
		rebuild formulae
	'''

	return df



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
			'Vwa':'vWA'
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


def get_header(file_path):
	header = None
	if file_path is not None:
		if file_path.endswith('.xlsx'):
			pass
		elif file_path.endswith('.xls') and os.path.isfile(file_path + 'x'):
			file_path = file_path + 'x'
		elif file_path.endswith('.xls'):
			file_path = convert_xls_to_xlsx(file_path)
		else:
			return header

		wb = openpyxl.load_workbook(file_path)
		ws = wb.worksheets[0]
		header = ws.oddHeader
		wb.close()
	return header

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
		# print(cell1.value, cell2.value)
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
		# print(cell1.value, cell2.value)
		cell2.value = formula_dict.get(cell1.value, cell2.value)
		# print(cell1.value, cell2.value)
		# print('')
	wb.save(filename)
	wb.close()


def main():
	owd = os.getcwd()  # original working directory
	os.chdir(r'X:\Hospital\Genetics Lab\DNA_Lab\Ghani')
	results = build_results_dict()
	f = [k[0] for k in results.keys()]
	f = set(f)
	for x in f:
		print(x)

	while True:
		build_profile_1(results)


if __name__ == '__main__':
	main()
