import re
import os
import sys
# import csv
import pandas as pd
import easygui
# from pprint import pprint
import openpyxl
from openpyxl.styles import Border, Side, PatternFill, Font, GradientFill, Alignment
import pyexcel
import string

pd.set_option('display.max_columns', 20)

def build_results_dict():
	results = {}
	result_csv = easygui.fileopenbox(msg='Select results file (should end in .csv)')
	if result_csv == None:
		exit()
	df = pd.read_csv(result_csv,delimiter='\s*,\s*', engine='python')
	# df = df.str.strip()
	df = df.iloc[:,[1,2,3,6]]
	df.dropna(how='any',inplace=True)
	df.reset_index(drop=True, inplace=True)
	keys = []
	for i in df.index:
		file_name = str(df.iloc[i]['Sample File Name']).upper()
		locus = str(df.iloc[i]['Marker']).upper()
		allele = str(df.iloc[i]['Allele']).upper()
		key = (file_name, locus, allele)
		keys.append(key)
		results[key] = df.iloc[i]['Area']
	assert len(keys) == len(set(keys))
	return results

def get_col_to_drop(df):
	col_to_drop = []
	for col in df.columns:
		if 'Unnamed' in col and df[col].isnull().all():
			col_to_drop.append(col)
	# if len(col_to_drop) > 4:
	# 	col_to_drop = col_to_drop[0:4]
	return col_to_drop

def build_profile(res):
	cases = sorted(list({k[0] for k in res.keys()}))
	choices = easygui.multchoicebox(msg='Pick cases that share a template', choices=cases, preselect=None)
	# print('choices = {}'.format(choices))
	if choices == None:
		exit()
	# col_rename = {
	# 	'Unnamed: 0': 'Locus',
	# 	'Unnamed: 2': '',
	# 	'Unnamed: 4': '',
	# 	'Unnamed: 7': '',
	# 	'Unnamed: 9': '',
	# 	'Unnamed: 12': '',
	# 	'Unnamed: 13': '',
	# 	'Unnamed: 15': '',
	# }

	msg = 'open template for ' + ', '.join(choices)
	template = easygui.fileopenbox(title=msg)
	if template == None:
		exit()
	for choice in choices:
		file_name = str(choice).upper()
		df = pd.read_excel(template)
		# temp_fname = 'temp1_' + file_name.replace('.FSA', '.xlsx')
		# df.to_excel(temp_fname, index=False)

		replacement_dict = {
			'THO1': 'TH01',
			'Amelogenin': 'AMEL',
			'amelogenin': 'AMEL',
			'AMELOGENIN': 'AMEL',
			'Recipient (Host) Alleles': 'Host'
		}
		df.replace(to_replace=replacement_dict, inplace=True)

		# drop up to four empty columns
		col_to_drop = get_col_to_drop(df)
		df.drop(axis=1, columns=col_to_drop, inplace=True)
		# temp_fname = 'temp2_' + file_name.replace('.FSA', '.xlsx')
		# df.to_excel(temp_fname, index=False)

		# get locations of 'Allele'
		allele_ij = []
		for i in df.index:
			for j,v in enumerate(df.iloc[i]):
				if v == 'Allele':
					allele_ij.append([i,j])
		# print(allele_ij)

		for i,j in allele_ij:
			locus = str(df.iloc[i,0]).upper()
			d1 = str(df.iloc[i,j+1]).upper()
			d2 = str(df.iloc[i,j+2]).upper()
			h1 = str(df.iloc[i,j+3]).upper()
			h2 = str(df.iloc[i,j+4]).upper()


			if len(d1) > 0 and d1!=pd.np.nan and d1!='NAN':
				key = (file_name, locus, d1)
				df.iat[i+1,j+1] = res.get(key,0)
			else:
				df.iat[i+1,j+1] = pd.np.nan

			if len(d2) > 0 and d2!=pd.np.nan and d2!='NAN':
				key = (file_name, locus, d2)
				df.iat[i+1,j+2] = res.get(key,0)
			else:
				df.iat[i+1,j+2] = pd.np.nan

			if len(h1) > 0 and h1!=pd.np.nan and h1!='NAN':
				key = (file_name, locus, h1)
				df.iat[i+1,j+3] = res.get(key,0)
			else:
				df.iat[i+1,j+3] = pd.np.nan

			if len(h2) > 0 and h2!=pd.np.nan and h2!='NAN':
				key = (file_name, locus, h2)
				df.iat[i+1,j+4] = res.get(key,0)
			else:
				df.iat[i+1,j+4] = pd.np.nan



		# for i in df.index:
		# 	# print(df.iloc[i])
		# 	j = [j for j in range(0,len(df.iloc[i])) if df.iloc[i,j] == 'Allele']
		# 	# print(i,j)
		# 	if df.iloc[i,7] == 'Allele':
		# 		locus = str(df.iloc[i,0]).upper()
		# 		d1 = str(df.iloc[i,8]).upper()
		# 		d2 = str(df.iloc[i,9]).upper()
		# 		h1 = str(df.iloc[i,12]).upper()
		# 		h2 = str(df.iloc[i,13]).upper()

		# 		if len(d1) > 0 and d1!=pd.np.nan and d1!='NAN':
		# 			key = (file_name, locus, d1)
		# 			df.iat[i+1,8] = res.get(key,0)
		# 		else:
		# 			df.iat[i+1,8] = pd.np.nan

		# 		if len(d2) > 0 and d2!=pd.np.nan and d2!='NAN':
		# 			key = (file_name, locus, d2)
		# 			df.iat[i+1,9] = res.get(key,0)
		# 		else:
		# 			df.iat[i+1,9] = pd.np.nan

		# 		if len(h1) > 0 and h1!=pd.np.nan and h1!='NAN':
		# 			key = (file_name, locus, h1)
		# 			df.iat[i+1,12] = res.get(key,0)
		# 		else:
		# 			df.iat[i+1,12] = pd.np.nan

		# 		if len(h2) > 0 and h2!=pd.np.nan and h2!='NAN':
		# 			key = (file_name, locus, h2)
		# 			df.iat[i+1,13] = res.get(key,0)
		# 		else:
		# 			df.iat[i+1,13] = pd.np.nan


		# Get rid of the remaining 'Unnamed: #' column labels
		df.rename(columns=lambda x: re.sub(r'Unnamed.*', '', x), inplace=True)

		# Write the output
		output_file_name = file_name.replace('.FSA', '.xlsx')
		df.to_excel(output_file_name,index=False)
		insert_formulae(output_file_name)
		fix_formatting(output_file_name)

	# return output_file_name

def border_add(border, top=None, right=None, left=None, bottom=None):
	if top == None:
		top = border.top
	if left == None:
		left = border.left
	if right == None:
		right = border.right
	if bottom == None:
		bottom = border.bottom
	# print(top, left, right, bottom)
	return openpyxl.styles.Border(top=top, left=left, right=right, bottom=bottom)

def fix_formatting(fname):
	print(fname)
	assert fname.endswith('.xlsx')
	thin = Side(border_style='thin')
	medium = Side(border_style='medium')
	wb = openpyxl.load_workbook(fname)
	ws = wb.worksheets[0]
	# row_count = ws.max_row
	# column_count = ws.max_column

	# by default make all cells horizontal='center
	cells = [ws[c+str(r)] for c in string.ascii_uppercase[0:ws.max_column] for r in range(1,ws.max_row+1)]
	for cell in cells:
		cell.alignment = Alignment(horizontal='center')

	# make first column bold and left aligned
	cells = [ws['A'+str(r)] for r in range(1,ws.max_row+1)]
	for cell in cells:
		cell.font = Font(bold=True)
		cell.alignment = Alignment(horizontal='left')

	# apply medium thickness to right border of cols A, C, E, H, N, O, P, S
	cols = ['A', 'C', 'E', 'F', 'H', 'J', 'K', 'L', 'M']
	cells = [ws[c+str(r)] for r in range(1,ws.max_row+1) for c in cols]
	for cell in cells:
		cell.border = border_add(cell.border, right=medium)

	# apply medium thickness to lower border of even numbered rows
	cols = string.ascii_uppercase[0:ws.max_column]
	cells = [ws[c+str(r)] for c in cols for r in range(1,ws.max_row+1) if r % 2==0]
	for cell in cells:
		cell.border = border_add(cell.border, bottom=medium)

	wb.save(fname)

def insert_formulae(fname):
	assert fname.endswith('.xlsx')
	wb = openpyxl.load_workbook(fname)
	ws = wb.worksheets[0]

	wb.save(fname)


def main():
	owd = os.getcwd()	# original working directory
	os.chdir('X:\Hospital\Genetics Lab\DNA_Lab\Ghani')
	results = build_results_dict()
	while True:
		build_profile(results)



if __name__ == '__main__':
	main()