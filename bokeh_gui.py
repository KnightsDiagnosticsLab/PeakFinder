from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models.widgets import FileInput, DataTable, DateFormatter, TableColumn, RadioButtonGroup, RadioGroup, Select, TextAreaInput, MultiSelect
from bokeh.models import TextInput, Button, ColumnDataSource, CDSView, IndexFilter, Panel, Tabs
from convert_fsa_to_csv import convert_file, convert_file_content
from os.path import basename

import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfilename

import pandas as pd
from fsa import use_csv_module
import copy
import openpyxl

from extract_from_genemapper import fix_formatting


results_files = []
source_cases = {}
df_cases = {}


def reduce_rows(df):
	df.dropna(axis=1, how='all', inplace=True)
	cols = ['Sample File Name', 'Marker','Allele', 'Area']
	df.dropna(axis=0, how='any', subset=cols, inplace=True)
	df.sort_values(by=cols, ascending=True, inplace=True)
	df = df.astype({'Area':'int',
					'Size':'float',
					'Height':'int',
					'Data Point':'int'})
	return df


def on_donor_change(attrname, old, new):
	global data_table, current_case
	# print(old, new)
	# print(data_table.source.selected.indices)
	newest = [c for c in new if c not in old]
	if len(newest) > 0:
		current_case = newest[0]
		data_table.source.data = source_cases[newest[0]].data
		data_table.source.selected.indices = source_cases[newest[0]].selected.indices[:]


def on_host_click(attrname, old, new):
	# print(new)
	global source_cases, data_table, current_case
	current_case = new
	# indices = df.index[df['Sample File Name'] == new].tolist()
	# view = CDSView(source=source, filters=[IndexFilter(indices)])
	# data_table.view = view
	# data_table.source.selected.indices = sorted(list(universal_selected))
	data_table.source.data = source_cases[new].data
	data_table.source.selected.indices = source_cases[new].selected.indices[:]
	# curdoc().add_root(tables[new])



def on_results_click():
	global source_cases, results_files, df, source, data_table

	root = tk.Tk()
	root.withdraw()		# hide the root tk window
	file_path = askopenfilename(filetypes = (('Text file', '*.txt'),
											('Comma Separated Values','*.csv'),
											('Tab Separated Values','*.tsv'),
										),
								title = 'Choose GeneMapper results file.'
							)
	root.destroy()
	if basename(file_path) not in results_files:
		results_files.append(basename(file_path))
		results_text.value = '\n'.join(results_files)

		df_new = use_csv_module(file_path)
		df_new = reduce_rows(df_new)

		case_names = sorted(list(set(df_new['Sample File Name'].to_list())))
		# if len(case_names) > 10:
		# 	print(case_names)
		# 	select_donor_case.size = len(case_names)
		# 	print('select_donor_case.size = {}'.format(select_donor_case.size))

		for case in case_names:
			df_copy = df_new.loc[df_new['Sample File Name'] == case].copy(deep=True)
			df_copy.reset_index(inplace=True, drop=True)
			df_copy['Selected'] = False
			indices = pre_selected_indices(df_copy)
			df_copy.loc[indices,'Selected'] = True
			df_cases[case] = df_copy
			# print(df_copy)
			source_cases[case] = ColumnDataSource(df_copy)
			source_cases[case].selected.indices = indices

		select_host_case.options = list(source_cases.keys())
		select_donor_case.options = list(source_cases.keys())

	if len(results_files) == 1:
		select_host_case.value = list(source_cases.keys())[0]


def pre_selected_indices(df):
	indices = []
	marker_area_max = {}
	markers = sorted(list(set(df['Marker'].tolist())))
	for marker in markers:
		df_marker = df.loc[df['Marker'] == marker]
		marker_area_max[marker] = df_marker['Area'].max()
	for i in df.index.tolist():
		marker = df.iloc[i].loc['Marker']
		area = df.iloc[i].loc['Area']
		if area >= 0.6 * marker_area_max[marker]:
			indices.append(i)
	return indices


def string_to_number(s):
	try:
		s = float(s)
	except:
		pass
	return s


def save_excel_file(file_path):
	markers = ['D8S1179', 'D21S11', 'D7S820', 'CSF1PO', 'D3S1358', 'TH01', 'D13S317', 'D16S539', 'D2S1338', 'D19S433', 'vWA', 'TPOX', 'D18S51', 'AMEL', 'D5S818', 'FGA']
	host_case = select_host_case.value
	df_host = df_cases[host_case]
	df_host = df_host.loc[df_host['Selected'] == True]

	donor_cases = select_donor_case.value
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
	ws.cell(row=1, column=2, value=host_case)
	for i, donor_case in enumerate(donor_cases):
		c = 4 + 2*i
		ws.cell(row=1, column=c, value=donor_case)
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
	wb.save(file_path)
	fix_formatting(file_path)
	print('Done saving {}'.format(file_path))


def on_export_template_click():
	root = tk.Tk()
	root.withdraw()		# hide the root tk window
	file_path = asksaveasfilename(defaultextension = '.xlsx',
									filetypes=[('Excel', '*.xlsx')],
									title = 'Save Template')
	root.destroy()
	export_text.value = basename(file_path)
	save_excel_file(file_path)


def on_selected_change(attrname, old, new):
	global current_case

	indices = data_table.source.selected.indices[:]
	source_cases[current_case].selected.indices = indices[:]
	df_cases[current_case].loc[:,'Selected'] = False
	df_cases[current_case].loc[indices,'Selected'] = True
	# print(df_cases[current_case])

select_results = Button(label='Add GeneMapper Results', button_type='success')
select_results.on_click(on_results_click)
results_text = TextAreaInput(value='<results file>', disabled=True, rows=5)


select_host_case = Select(title='Select Host', options=list(source_cases.keys()))
select_host_case.on_change('value', on_host_click)


select_donor_case = MultiSelect(title='Select Donor(s) <ctrl+click to multiselect>',
								options=list(source_cases.keys()), size=20)
select_donor_case.on_change('value', on_donor_change)


export_template = Button(label='Export Template', button_type='warning')
export_template.on_click(on_export_template_click)
export_text = TextInput(value='<template file>', disabled=True)


enter_host_name = TextInput(title='Enter Host Name', value='<type host name here>')

columns = [
			TableColumn(field='Sample File Name', title='Sample File Name', width=300),
			TableColumn(field='Marker', title='Marker', width=75),
			TableColumn(field='Allele', title='Allele', width=50),
			TableColumn(field='Area', title='Area', width=50),
			]

col_temp = [
			TableColumn(field='Marker', title='Marker', width=75),
			TableColumn(field='Allele_1', title='Allele_1', width=50),
			TableColumn(field='Allele_2', title='Allele_2', width=50),
			]


source = ColumnDataSource()
source.selected.on_change('indices', on_selected_change)
data_table = DataTable(columns=columns, source=source, selectable='checkbox')
template_table = DataTable(source=ColumnDataSource())

col_0 = column(enter_host_name, select_results, results_text, select_host_case, select_donor_case, export_template)
col_1 = column(data_table, sizing_mode='stretch_height')

curdoc().add_root(row(col_0, col_1, sizing_mode='stretch_height'))
curdoc().title = 'PTE'


''' Outline of functions
	Input box for Host's name
	on_case_change
		DONE: add logic of pre-selecting certain rows
		color code rows by Marker -> harder than it should be!
		show live export template
		
'''