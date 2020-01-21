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

# df = pd.DataFrame()
results_files = []
cases_source = {}
# tables = {}
# universal_selected = set()


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
		# indices = df.index[df['Sample File Name'] == newest[0]].tolist()
		# view = CDSView(source=source, filters=[IndexFilter(indices)])
		# data_table.view = view
		# data_table.source.selected.indices = sorted(list(universal_selected))
		# data_table = tables[newest[0]]
		data_table.source.data = cases_source[newest[0]].data
		data_table.source.selected.indices = cases_source[newest[0]].selected.indices[:]


def on_host_change(attrname, old, new):
	# print(new)
	global cases_source, data_table, current_case
	current_case = new
	# indices = df.index[df['Sample File Name'] == new].tolist()
	# view = CDSView(source=source, filters=[IndexFilter(indices)])
	# data_table.view = view
	# data_table.source.selected.indices = sorted(list(universal_selected))
	data_table.source.data = cases_source[new].data
	data_table.source.selected.indices = cases_source[new].selected.indices[:]
	# curdoc().add_root(tables[new])



def on_results_change():
	global cases_source, results_files, df, source, data_table, results

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
		# df_new.index.name = 'Index'
		df_new = reduce_rows(df_new)
		# df = pd.concat([df, df_new], axis=0, sort=False, ignore_index=True)
	
		# source = ColumnDataSource(df)
		# data_table.source.data = source.data		# This will probably reset the selected indices

		case_names = sorted(list(set(df_new['Sample File Name'].to_list())))

		for case in case_names:
			df_copy = df_new.loc[df_new['Sample File Name'] == case].copy(deep=True)
			df_copy.reset_index(inplace=True, drop=True)
			indices = pre_selected_indices(df_copy)
			cases_source[case] = ColumnDataSource(df_copy)
			cases_source[case].selected.indices = indices
			# cases_source[case].selected.on_change('indices', on_selected_change)
			# table = DataTable(columns=columns, source=cases_source[case], selectable='checkbox')
			# table.source.selected.on_change('indices', on_selected_change)
			# tables[case] = DataTable(columns=columns, source=cases_source[case], selectable='checkbox')
			# tables[case] = table


		select_host_case.options = list(cases_source.keys())
		select_donor_case.options = list(cases_source.keys())

	if len(results_files) == 1:
		select_host_case.value = list(cases_source.keys())[0]


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
		# print(marker)
	# print(df)
	# for row in df:
	# 	print(row)
	return indices


def on_export_template_change():
	root = tk.Tk()
	root.withdraw()		# hide the root tk window
	file_path = asksaveasfilename(defaultextension = '.xlsx',
									filetypes=[('Excel', '*.xlsx')],
									title = 'Save Template')
	root.destroy()
	export_text.value = basename(file_path)


def on_selected_change(attrname, old, new):
	global current_case
	# print('\tfrom inside on_selected_change')
	# print('\t\tdata_table.source.selected.indices = {}'.format(data_table.source.selected.indices))
	cases_source[current_case].selected.indices = data_table.source.selected.indices[:]
# 	global universal_selected
# 	print('data_table.')
# 	print('BEFORE universal_selected = {}'.format(universal_selected))
# 	old_set = set(old)
# 	old_set.discard(None)
# 	print('old_set = {}'.format(old_set))
# 	new_set = set(new)
# 	new_set.discard(None)
# 	print('new_set = {}'.format(new_set))
# 	universal_selected = universal_selected - old_set
# 	universal_selected = universal_selected | new_set
# 	print('AFTER universal_selected = {}\n'.format(universal_selected))

select_results = Button(label='Add GeneMapper Results', button_type='success')
select_results.on_click(on_results_change)
results_text = TextAreaInput(value='<results file>', disabled=True, rows=5)

select_host_case = Select(title='Select Host', options=list(cases_source.keys()))
select_host_case.on_change('value', on_host_change)

select_donor_case = MultiSelect(title='Select Donor(s) <ctrl+click to multiselect>',
								options=list(cases_source.keys()), size=12)
select_donor_case.on_change('value', on_donor_change)

export_template = Button(label='Export Template', button_type='success')
export_template.on_click(on_export_template_change)
export_text = TextInput(value='<template file>', disabled=True)


columns = [
			TableColumn(field='Sample File Name', title='Sample File Name', width=300),
			TableColumn(field='Marker', title='Marker', width=75),
			TableColumn(field='Allele', title='Allele', width=50),
			TableColumn(field='Area', title='Area', width=50),
			]


source = ColumnDataSource()
source.selected.on_change('indices', on_selected_change)
data_table = DataTable(columns=columns, source=source, selectable='checkbox')


grid = gridplot(
				[[select_results, results_text],
				[None, select_host_case],
				[None, select_donor_case]],
				merge_tools=True,
				toolbar_options=dict(logo=None)
			)

curdoc().add_root(grid)
curdoc().add_root(column(data_table, sizing_mode='stretch_height'))
curdoc().title = 'PTE'


''' Outline of functions
	Input box for Host's name
	on_case_change
		DONE: add logic of pre-selecting certain rows
		color code rows by Marker -> harder than it should be!
		show live export template
		
'''