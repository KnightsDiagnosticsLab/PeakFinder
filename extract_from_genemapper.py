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
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 50)


def make_results_dict_from_template(template):
	# print('Now inside makeshift_results_dictionary')
	results_dict = {}
	host_case = None
	donor_case = None
	host_cell = None
	donor_cell = None

	if isinstance(template, pd.DataFrame):
		wb = df_to_wb(template)
	elif isinstance(template, openpyxl.Workbook):
		pass
	elif os.path.isfile(str(template)) and template.endswith('.xlsx'):
		# print(template_path)
		wb = openpyxl.load_workbook(template)
		ws = wb.worksheets[0]
	elif os.path.isfile(str(template)) and template.endswith('.xls'):
		template = convert_xls_to_xlsx(template)
		wb = openpyxl.load_workbook(template)

	ws = wb.worksheets[0]

	replacement_dict_no_regex = {
		'THO1': 'TH01',
		'Recipient (Host) Alleles': 'Host',
	}
	replacement_dict_yes_regex = {
		'[vV][wW][aA]':'vWA',
		'[aA][mM][eE][lL][oO][gG][eE][nN][iI][nN]':'AMEL',
	}

	ws = replace_cell_values(ws, replacement_dict_no_regex, regex=False)
	ws = replace_cell_values(ws, replacement_dict_yes_regex, regex=True)

	df = pd.DataFrame(ws.values)
	# print(df)
	old = True
	host_cell = cell_with_value(ws, 'ENG Host:')
	donor_cell = cell_with_value(ws, 'DEG Donor:')
	if host_cell is None:
		old = False
		host_cell = cell_with_value(ws, 'Host')
		donor_cell = cell_with_value(ws, 'Donor')

	if host_cell is not None and donor_cell is not None:

		if old:
			host_case = str(ws.cell(row=host_cell.row + 1, column=host_cell.column).value)
			donor_case = str(ws.cell(row=donor_cell.row + 1, column=donor_cell.column).value)
		else:
			host_case = str(ws.cell(row=host_cell.row - 1, column=host_cell.column).value)
			donor_case = str(ws.cell(row=donor_cell.row - 1, column=donor_cell.column).value)

		markers = ['D8S1179', 'D21S11', 'D7S820', 'CSF1PO', 'D3S1358', 'TH01', 'D13S317', 'D16S539', 'D2S1338', 'D19S433', 'vWA', 'TPOX', 'D18S51', 'AMEL', 'D5S818', 'FGA']
		marker_cells = [cell_with_value(ws, marker) for marker in markers if cell_with_value(ws, marker) is not None]

		for marker_cell in marker_cells:
			marker = str(marker_cell.value)
			# print('now looping through marker cells')
			# print(marker, marker_cell.value)
			host_allele_0 = ws.cell(row=marker_cell.row, column=host_cell.column).value
			host_allele_1 = ws.cell(row=marker_cell.row, column=host_cell.column + 1).value
			donor_allele_0 = ws.cell(row=marker_cell.row, column=donor_cell.column).value
			donor_allele_1 = ws.cell(row=marker_cell.row, column=donor_cell.column + 1).value

			if host_allele_0 is not None:
				results_dict[('Host', marker, host_allele_0)] = None
			if host_allele_1 is not None:
				results_dict[('Host', marker, host_allele_1)] = None
			if donor_allele_0 is not None:
				results_dict[('Donor', marker, donor_allele_0)] = None
			if donor_allele_1 is not None:
				results_dict[('Donor', marker, donor_allele_1)] = None

		wb.close()
	# print('Now looping through results_dict dictionary')
	# for k,v in results_dict.items():
	# 	print(k,v)
	# print('host_case = {}'.format(host_case))
	# print('donor_case = {}'.format(donor_case))
	return results_dict, host_case, donor_case


def make_template_from_existing_template(template):

	''' Helper function '''
	def string_to_number(s):
		try:
			s = float(s)
		except:
			pass
		return s

	''' Helper function '''
	def format_value(cell_value):
		if cell_value is not None:
			cell_value_formatted = str(cell_value).replace('.0', '')
		else:
			cell_value_formatted = None
		return cell_value_formatted


	results_dict, host_case, donor_case = make_results_dict_from_template(template)

	markers = ['D8S1179', 'D21S11', 'D7S820', 'CSF1PO', 'D3S1358', 'TH01', 'D13S317', 'D16S539', 'D2S1338', 'D19S433', 'vWA', 'TPOX', 'D18S51', 'AMEL', 'D5S818', 'FGA']

	wb = openpyxl.Workbook()
	ws = wb.active

	ws.cell(row=2, column=1).value = 'Marker'
	ws.cell(row=2, column=2).value = 'Host'
	ws.cell(row=2, column=4).value = 'Donor'

	host_case_abbrev = re.sub(r'_PTE.*$', '', str(host_case))
	donor_case_abbrev = re.sub(r'_PTE.*$', '', str(donor_case))

	ws.cell(row=1, column=2).value = host_case_abbrev
	ws.cell(row=1, column=4, value=donor_case_abbrev)

	'''	Markers & Alleles
	'''
	for i, marker in enumerate(markers):

		r = 1+(i+1)*2
		ws.cell(row=r, column=1).value = marker

		alleles = [k[2] for k in results_dict.keys() if k[0] == 'Host' and k[1] == marker]

		for j, allele in enumerate(alleles):
			# print(marker, j, allele)
			allele = string_to_number(allele)
			c = j+2
			ws.cell(row=r, column=c).value = allele

		alleles = [k[2] for k in results_dict.keys() if k[0] == 'Donor' and k[1] == marker]

		for j, allele in enumerate(alleles):
			allele = string_to_number(allele)
			c = 4 + j
			ws.cell(row=r, column=c).value = allele

	''' Add in column of Allele/Area
	'''
	caa = 6
	for i, marker in enumerate(markers):
		r1 = 1+(i+1)*2
		r2 = r1 + 1
		ws.cell(row=r1, column=caa).value = 'Allele'
		ws.cell(row=r2, column=caa).value = 'Area'

	''' Copy columns
	'''
	for i in range(2,caa,2):
		for r in range(2,ws.max_row + 1):
			c1 = 2*caa - (i+1)
			c2 = c1 + 1
			ws.cell(row=r, column=c1, value=ws.cell(row=r, column=i).value)
			ws.cell(row=r, column=c2, value=ws.cell(row=r, column=i+1).value)

	'''	Add in % Host & Forumula columns
	'''
	ws.cell(row=2, column=2*caa - 1, value='% Host')
	ws.cell(row=2, column=2*caa, value='Formula')
	ws.cell(row=ws.max_row+1, column=2*caa-2, value='%Host')
	ws.cell(row=ws.max_row+1, column=2*caa-2, value='%Donor')

	'''	Add the actual formulae. For now only if there's one donor, because
		I haven't been given a cheatsheet for 2+ donors.
	'''
	percent_host_col = 2*caa - 1		# col num where formula goes. Actual Excel formula, not the text version.
	d1_col = percent_host_col - 4		# col num for donor allele 1
	d2_col = percent_host_col - 3		# col num for donor allele 2
	h1_col = percent_host_col - 2		# col num for host allele 1
	h2_col = percent_host_col - 1		# col num for host allele 2
	for r2 in range(4, 4+2*len(markers), 2):
		r1 = r2 - 1

		d1_alle = ws.cell(row=r1, column=d1_col)
		d2_alle = ws.cell(row=r1, column=d2_col)
		h1_alle = ws.cell(row=r1, column=h1_col)
		h2_alle = ws.cell(row=r1, column=h2_col)

		d1_alle_val = format_value(d1_alle.value)
		d2_alle_val = format_value(d2_alle.value)
		h1_alle_val = format_value(h1_alle.value)
		h2_alle_val = format_value(h2_alle.value)

		d1_area = ws.cell(row=r2, column=d1_col)
		d2_area = ws.cell(row=r2, column=d2_col)
		h1_area = ws.cell(row=r2, column=h1_col)
		h2_area = ws.cell(row=r2, column=h2_col)

		f1 = ws.cell(row=r1, column=percent_host_col+1)
		f1.alignment = Alignment(horizontal='center')
		f2 = ws.cell(row=r2, column=percent_host_col+1)
		f2.alignment = Alignment(horizontal='left')

		d = {d1_alle_val, d2_alle_val}
		d.discard(None)
		h = {h1_alle_val, h2_alle_val}
		h.discard(None)

		percent_host = ws.cell(row=r2, column=percent_host_col)

		# print('d = {}, h = {}'.format(d,h))

		if len(d | h) == 4:
			f1.value = '{} + {}'.format(h1_alle_val,
										h2_alle_val)
			f1.alignment = Alignment(horizontal='center')

			f2.value = '{} + {} + {} + {}'.format(h1_alle_val,
													h2_alle_val,
													d1_alle_val,
													d2_alle_val)

			percent_host.value = '=100*SUM({}:{})/SUM({},{},{},{})'.format(h1_area.coordinate,
																h2_area.coordinate,
																d1_area.coordinate,
																d2_area.coordinate,
																h1_area.coordinate,
																h2_area.coordinate)

		if len(h) == 1 and len(h | d) == 3:
			f1.value = '{}'.format(h1_alle_val)
			f1.alignment = Alignment(horizontal='center')

			f2.value = '{} + {} + {}'.format(h1_alle_val,
											d1_alle_val,
											d2_alle_val)

			percent_host.value = '=100*{}/SUM({},{},{})'.format(h1_area.coordinate,
														d1_area.coordinate,
														d2_area.coordinate,
														h1_area.coordinate)

		if len(d) == 1 and len( h | d) == 3:
			f1.value = '{} + {}'.format(h1_alle_val,
										h2_alle_val)
			f1.alignment = Alignment(horizontal='center')

			f2.value = '{} + {} + {}'.format(h1_alle_val,
											h2_alle_val,
											d1_alle_val)

			percent_host.value = '=100*SUM({}:{})/SUM({},{},{})'.format(h1_area.coordinate,
																h2_area.coordinate,
																d1_area.coordinate,
																h1_area.coordinate,
																h2_area.coordinate)

		if len(h) == 1 and len(d) == 1 and len(h | d) == 2:
			f1.value = '{}'.format(h1_alle_val)
			f1.alignment = Alignment(horizontal='center')

			f2.value = '{} + {}'.format(h1_alle_val,
										d1_alle_val)
			f2.alignment = Alignment(horizontal='center')

			percent_host = '={}/SUM({},{})'.format(h1_area.coordinate,
												d1_area.coordinate,
												h1_area.coordinate)

		if len(h) == 2 and len(d) == 2 and len(h | d) == 3:
			if h1_alle_val not in d:
				h_unique_alle_val = h1_alle_val
				h_unique_area = h1_area
			else:
				h_unique_alle_val = h2_alle_val
				h_unique_area = h2_area

			if d1_alle_val not in h:
				d_unique_alle_val = d1_alle_val
				d_unique_area = d1_area
			else:
				d_unique_alle_val = d2_alle_val
				d_unique_area = d2_area

			f1.value = '{}'.format(h_unique_alle_val)
			f1.alignment = Alignment(horizontal='center')

			f2.value = '{} + {}'.format(h_unique_alle_val,
										d_unique_alle_val)
			f2.alignment = Alignment(horizontal='center')

			percent_host.value = '=100*{}/SUM({},{})'.format(h_unique_area.coordinate,
												d_unique_area.coordinate,
												h_unique_area.coordinate)

		if len(h) == 1 and len(d) == 2 and len(h | d) == 2:
			if d1_alle_val in h:
				A_alle_val = d2_alle_val
				A_area = d2_area
			else:
				A_alle_val = d1_alle_val
				A_area = d1_area

			if h1_alle_val in d:
				H_alle_val = h1_alle_val
				H_area = h1_area
			else:
				H_alle_val = h2_alle_val
				H_area = h2_area

			f2.value = '1 - (2x{}/({} + {}))'.format(A_alle_val,
													A_alle_val,
													H_alle_val)

			percent_host.value = '=100*(1-(2*{}/({}+{})))'.format(A_area.coordinate,
																A_area.coordinate,
																H_area.coordinate)
		
		if len(h) == 2 and len(d) == 1 and len(h | d) == 2:
			if h1_alle_val in d:
				A_alle_val = h2_alle_val
				A_area = h2_area
			else:
				A_alle_val = h1_alle_val
				A_area = h1_area

			if d1_alle_val in h:
				D_alle_val = d1_alle_val
				D_area = d1_area
			else:
				D_alle_val = d2_alle_val
				D_area = d2_area

			f1.value = '2x{}'.format(A_alle_val)
			f1.alignment = Alignment(horizontal='center')

			f2.value = '{} + {}'.format(A_alle_val,
										D_alle_val)
			f2.alignment = Alignment(horizontal='center')

			percent_host.value = '=100*(2*{})/({}+{})'.format(A_area.coordinate,
																A_area.coordinate,
																D_area.coordinate)

		'''	Formula for average of Percent Host column
		'''
		percent_host_avg = ws.cell(row=3+2*len(markers), column=percent_host_col)
		start = ws.cell(row=3, column=percent_host_col)
		end = ws.cell(row=2+2*len(markers), column=percent_host_col)
		percent_host_avg.value = '=AVERAGE({}:{})'.format(start.coordinate, end.coordinate)

		percent_donor_avg = ws.cell(row=4+2*len(markers), column=percent_host_col)
		percent_donor_avg.value = '=100-{}'.format(percent_host_avg.coordinate)

	# df = pd.DataFrame(ws.values)
	# print(df)

	wb.close()

	return wb

# def fill_in_results(template, results_dict):
# 	return df

def build_profile_2(template, sample_name='', res={}):
	# print('Now running build_profile_2')

	# pprint(res)
	df = pd.DataFrame()
	if isinstance(template, pd.DataFrame):
		df = template.copy(deep=True)
	elif isinstance(template, openpyxl.Workbook):
		ws = template.worksheets[0]
		df = pd.DataFrame(ws.values)
	elif os.path.isfile(str(template)):
		if template.endswith('.xls'):
			template = convert_xls_to_xlsx(template)
		if template.endswith('.xlsx'):
			# print(template_path)
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

			wb = openpyxl.load_workbook(template)
			ws = wb.worksheets[0]

			df = pd.DataFrame(ws.values)
	else:
		return df	# return an empty dataframe

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

	# print(df)

	'''	replace cells that reference other cells with the ref cell's value '''
	df = replace_cell_ref_with_value(df)
	# for i, j in allele_ij:
	# 	locus = str(df.iloc[i, 0])
	# 	for k in range(1, j):
	# 		x = str(df.iloc[i, j + k])
	# 		if x.startswith('='):
	# 			coor_ws_text = x.replace('=','')
	# 			coor_ws_num = coordinate_from_string(coor_ws_text)
	# 			coor_idx = column_index_from_string(coor_ws_num[0]) - 1
	# 			coor_col = coor_ws_num[1] - 1
	# 			print('i = {}, j = {}, k = {}, x = {}, coor_ws_text = {}, coor_idx = {}, coor_col = {}'. format(i,j,k,x,coor_ws_text,coor_idx,coor_col))
	# 			new_val =  df.iloc[coor_idx, coor_col]
	# 			df.iloc[i, j + k] = df.iloc[coor_idx, coor_col]

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
	# wb = df_to_wb(df)

	# print(df)
	return df






def remake_template_wb(host_case, donor_cases, df_cases):

	''' Helper function '''
	def string_to_number(s):
		try:
			s = float(s)
		except:
			pass
		return s

	''' Helper function '''
	def format_value(cell_value):
		if cell_value is not None:
			cell_value_formatted = str(cell_value).replace('.0', '')
		else:
			cell_value_formatted = None
		return cell_value_formatted


	markers = ['D8S1179', 'D21S11', 'D7S820', 'CSF1PO', 'D3S1358', 'TH01', 'D13S317', 'D16S539', 'D2S1338', 'D19S433', 'vWA', 'TPOX', 'D18S51', 'AMEL', 'D5S818', 'FGA']
	# host_case = select_host_case.value
	df_host = df_cases[host_case]
	df_host = df_host.loc[df_host['Selected'] == True]
	# print('df_host')
	# print(df_host)

	# donor_cases = select_donor_cases.value
	donors = {}
	for donor_case in donor_cases:
		df_donor = df_cases[donor_case]
		df_donor = df_donor.loc[df_donor['Selected'] == True]
		donors[donor_case] = df_donor.copy(deep=True)

	wb = openpyxl.Workbook()
	ws = wb.active
	'''	Header, etc.
	'''
	ws.oddHeader.center.text = enter_host_name.value
	# ws.oddHeader.center.size = 14
	ws.cell(row=2, column=1, value='Marker')
	ws.cell(row=2, column=2, value='Host')
	host_case_abbrev = re.sub(r'_PTE.*$', '', host_case)
	ws.cell(row=1, column=2, value=host_case_abbrev)
	for i, donor_case in enumerate(donor_cases):
		donor_case_abbrev = re.sub(r'_PTE.*$', '', donor_case)
		c = 4 + 2*i
		ws.cell(row=1, column=c, value=donor_case_abbrev)
		if len(donor_cases) == 1:
			donor_num = 'Donor'
		else:
			donor_num = 'Donor ' + str(i+1)
		ws.cell(row=2, column=c, value=donor_num)

	'''	Markers & Alleles
	'''
	for i, marker in enumerate(markers):
		r = 1+(i+1)*2
		df_marker = df_host.loc[df_host['Marker'] == marker]
		ws.cell(row=r, column=1, value=marker)
		# print(df_marker)
		for j, allele in enumerate(df_marker['Allele']):
			# print(marker, j, allele)
			allele = string_to_number(allele)
			c = j+2
			ws.cell(row=r, column=c, value=allele)
		
		for j, donor in enumerate(donors.keys()):
			df_donor = donors[donor]
			df_marker = df_donor.loc[df_donor['Marker'] == marker]
			for k, allele in enumerate(df_marker['Allele']):
				allele = string_to_number(allele)
				c = (4 + 2*j) + k
				ws.cell(row=r, column=c, value=allele)

	''' Add in column of Allele/Area
	'''
	caa = 2*len(donors.keys()) + 4
	for i, marker in enumerate(markers):
		r1 = 1+(i+1)*2
		r2 = r1 + 1
		ws.cell(row=r1, column=caa, value='Allele')
		ws.cell(row=r2, column=caa, value='Area')

	''' Copy columns
	'''
	for i in range(2,caa,2):
		for r in range(2,ws.max_row + 1):
			c1 = 2*caa - (i+1)
			c2 = c1 + 1
			ws.cell(row=r, column=c1, value=ws.cell(row=r, column=i).value)
			ws.cell(row=r, column=c2, value=ws.cell(row=r, column=i+1).value)

	'''	Add in % Host & Forumula columns
	'''
	ws.cell(row=2, column=2*caa - 1, value='% Host')
	ws.cell(row=2, column=2*caa, value='Formula')
	ws.cell(row=ws.max_row+1, column=2*caa-2, value='%Host')
	ws.cell(row=ws.max_row+1, column=2*caa-2, value='%Donor')


	if len(donor_cases) == 1:

		'''	Add the actual formulae. For now only if there's one donor, because
			I haven't been given a cheatsheet for 2+ donors.
		'''
		percent_host_col = 2*caa - 1		# col num where formula goes. Actual Excel formula, not the text version.
		d1_col = percent_host_col - 4		# col num for donor allele 1
		d2_col = percent_host_col - 3		# col num for donor allele 2
		h1_col = percent_host_col - 2		# col num for host allele 1
		h2_col = percent_host_col - 1		# col num for host allele 2
		for r2 in range(4, 4+2*len(markers), 2):
			r1 = r2 - 1

			d1_alle = ws.cell(row=r1, column=d1_col)
			d2_alle = ws.cell(row=r1, column=d2_col)
			h1_alle = ws.cell(row=r1, column=h1_col)
			h2_alle = ws.cell(row=r1, column=h2_col)

			d1_alle_val = format_value(d1_alle.value)
			d2_alle_val = format_value(d2_alle.value)
			h1_alle_val = format_value(h1_alle.value)
			h2_alle_val = format_value(h2_alle.value)

			d1_area = ws.cell(row=r2, column=d1_col)
			d2_area = ws.cell(row=r2, column=d2_col)
			h1_area = ws.cell(row=r2, column=h1_col)
			h2_area = ws.cell(row=r2, column=h2_col)

			f1 = ws.cell(row=r1, column=percent_host_col+1)
			f1.alignment = Alignment(horizontal='center')
			f2 = ws.cell(row=r2, column=percent_host_col+1)
			f2.alignment = Alignment(horizontal='left')

			d = {d1_alle_val, d2_alle_val}
			d.discard(None)
			h = {h1_alle_val, h2_alle_val}
			h.discard(None)

			percent_host = ws.cell(row=r2, column=percent_host_col)

			# print('d = {}, h = {}'.format(d,h))

			if len(d | h) == 4:
				f1.value = '{} + {}'.format(h1_alle_val,
											h2_alle_val)
				f1.alignment = Alignment(horizontal='center')

				f2.value = '{} + {} + {} + {}'.format(h1_alle_val,
														h2_alle_val,
														d1_alle_val,
														d2_alle_val)

				percent_host.value = '=100*SUM({}:{})/SUM({},{},{},{})'.format(h1_area.coordinate,
																	h2_area.coordinate,
																	d1_area.coordinate,
																	d2_area.coordinate,
																	h1_area.coordinate,
																	h2_area.coordinate)

			if len(h) == 1 and len(h | d) == 3:
				f1.value = '{}'.format(h1_alle_val)
				f1.alignment = Alignment(horizontal='center')

				f2.value = '{} + {} + {}'.format(h1_alle_val,
												d1_alle_val,
												d2_alle_val)

				percent_host.value = '=100*{}/SUM({},{},{})'.format(h1_area.coordinate,
															d1_area.coordinate,
															d2_area.coordinate,
															h1_area.coordinate)

			if len(d) == 1 and len( h | d) == 3:
				f1.value = '{} + {}'.format(h1_alle_val,
											h2_alle_val)
				f1.alignment = Alignment(horizontal='center')

				f2.value = '{} + {} + {}'.format(h1_alle_val,
												h2_alle_val,
												d1_alle_val)

				percent_host.value = '=100*SUM({}:{})/SUM({},{},{})'.format(h1_area.coordinate,
																	h2_area.coordinate,
																	d1_area.coordinate,
																	h1_area.coordinate,
																	h2_area.coordinate)

			if len(h) == 1 and len(d) == 1 and len(h | d) == 2:
				f1.value = '{}'.format(h1_alle_val)
				f1.alignment = Alignment(horizontal='center')

				f2.value = '{} + {}'.format(h1_alle_val,
											d1_alle_val)
				f2.alignment = Alignment(horizontal='center')

				percent_host = '={}/SUM({},{})'.format(h1_area.coordinate,
													d1_area.coordinate,
													h1_area.coordinate)

			if len(h) == 2 and len(d) == 2 and len(h | d) == 3:
				if h1_alle_val not in d:
					h_unique_alle_val = h1_alle_val
					h_unique_area = h1_area
				else:
					h_unique_alle_val = h2_alle_val
					h_unique_area = h2_area

				if d1_alle_val not in h:
					d_unique_alle_val = d1_alle_val
					d_unique_area = d1_area
				else:
					d_unique_alle_val = d2_alle_val
					d_unique_area = d2_area

				f1.value = '{}'.format(h_unique_alle_val)
				f1.alignment = Alignment(horizontal='center')

				f2.value = '{} + {}'.format(h_unique_alle_val,
											d_unique_alle_val)
				f2.alignment = Alignment(horizontal='center')

				percent_host.value = '=100*{}/SUM({},{})'.format(h_unique_area.coordinate,
													d_unique_area.coordinate,
													h_unique_area.coordinate)

			if len(h) == 1 and len(d) == 2 and len(h | d) == 2:
				if d1_alle_val in h:
					A_alle_val = d2_alle_val
					A_area = d2_area
				else:
					A_alle_val = d1_alle_val
					A_area = d1_area

				if h1_alle_val in d:
					H_alle_val = h1_alle_val
					H_area = h1_area
				else:
					H_alle_val = h2_alle_val
					H_area = h2_area

				f2.value = '1 - (2x{}/({} + {}))'.format(A_alle_val,
														A_alle_val,
														H_alle_val)

				percent_host.value = '=100*(1-(2*{}/({}+{})))'.format(A_area.coordinate,
																	A_area.coordinate,
																	H_area.coordinate)
			
			if len(h) == 2 and len(d) == 1 and len(h | d) == 2:
				if h1_alle_val in d:
					A_alle_val = h2_alle_val
					A_area = h2_area
				else:
					A_alle_val = h1_alle_val
					A_area = h1_area

				if d1_alle_val in h:
					D_alle_val = d1_alle_val
					D_area = d1_area
				else:
					D_alle_val = d2_alle_val
					D_area = d2_area

				f1.value = '2x{}'.format(A_alle_val)
				f1.alignment = Alignment(horizontal='center')

				f2.value = '{} + {}'.format(A_alle_val,
											D_alle_val)
				f2.alignment = Alignment(horizontal='center')

				percent_host.value = '=100*(2*{})/({}+{})'.format(A_area.coordinate,
																	A_area.coordinate,
																	D_area.coordinate)

		'''	Formula for average of Percent Host column
		'''
		percent_host_avg = ws.cell(row=3+2*len(markers), column=percent_host_col)
		start = ws.cell(row=3, column=percent_host_col)
		end = ws.cell(row=2+2*len(markers), column=percent_host_col)
		percent_host_avg.value = '=AVERAGE({}:{})'.format(start.coordinate, end.coordinate)

		percent_donor_avg = ws.cell(row=4+2*len(markers), column=percent_host_col)
		percent_donor_avg.value = '=100-{}'.format(percent_host_avg.coordinate)


	# if file_path is not None:
	# 	if file_path.endswith('.xlsx'):
	# 		'''	Save the file before running fix_formatting '''
	# 		wb.save(file_path)
	# 		wb.close()

	# 		'''	Fix formatting '''
	# 		fix_formatting(file_path)
	# 		print('Done saving {}'.format(file_path))

	wb.close()

	return wb


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
		# print('case_name = {}'.format(case_name))
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


# def main():
# 	owd = os.getcwd()  # original working directory
# 	os.chdir(r'X:\Hospital\Genetics Lab\DNA_Lab\Ghani')
# 	results = build_results_dict()
# 	f = [k[0] for k in results.keys()]
# 	f = set(f)
# 	for x in f:
# 		print(x)

# 	while True:
# 		build_profile_1(results)


# if __name__ == '__main__':
# 	main()