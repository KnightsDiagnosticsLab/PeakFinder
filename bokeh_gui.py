from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models.widgets import FileInput, DataTable, DateFormatter, TableColumn, RadioButtonGroup, RadioGroup, Select, TextAreaInput
from bokeh.models import TextInput, Button, ColumnDataSource
from convert_fsa_to_csv import convert_file, convert_file_content
from os.path import basename
# import easygui
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfilename
import pandas as pd
from fsa import use_csv_module

df = pd.DataFrame()
data = pd.DataFrame(columns=['Sample File Name','Marker','Allele','Area'])
results_files = []
cases = []

def select_host_callback():
	fpath = convert_file()
	host_text.value = basename(fpath)

select_host = Button(label='Select Host', button_type='success')
select_host.on_click(select_host_callback)
host_text = TextInput(value='<host file>', disabled=True)
host_row = row(select_host, host_text)

def select_donor_callback():
	fpath = convert_file()
	donor_text.value = basename(fpath)

select_donor = Button(label='Select Donor', button_type='success')
select_donor.on_click(select_donor_callback)
donor_text = TextInput(value='<donor file>', disabled=True)
donor_row = row(select_donor, donor_text)

def reduce_rows(df):
	cols = ['Sample File Name', 'Marker','Allele', 'Area']
	data = df[cols].dropna(how='any')
	data.sort_values(by=cols, ascending=True, inplace=True)
	return data

def select_case_callback(attrname, old, new):
	global df
	data = df.loc[df['Sample File Name'] == new]
	source = ColumnDataSource(data)
	# columns = [TableColumn(field='Sample File Name', title='Sample File Name', width=300),
	# 			TableColumn(field='Marker', title='Marker', width=75),
	# 			TableColumn(field='Allele', title='Allele', width=50),
	# 			TableColumn(field='Area', title='Area', width=50),
	# 			]
	data_table.source.data = source.data

select_case = Select(options=cases)
select_case.on_change('value', select_case_callback)
# data_table = DataTable(source=source, columns=columns)
# curdoc().add_root()

def select_results_callback():
	global cases, results_files, df

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
		df = pd.concat([df, df_new], axis=0, sort=True)
		df = reduce_rows(df)

		cases = sorted(list(set(df['Sample File Name'].to_list())))
		select_case.options = cases
		select_case.value = cases[0]

	# data = df.loc[df['Sample File Name'] == select_case.value]
	# source = ColumnDataSource(data)
	# data_table.source.data = source.data

# select_results = FileInput(accept='.csv, .txt, .tsv')

select_results = Button(label='Add GeneMapper Results', button_type='success')
select_results.on_click(select_results_callback)
results_text = TextAreaInput(value='<results file>', disabled=True, rows=5)
results_row = row(select_results, results_text)


def export_template_callback():
	root = tk.Tk()
	root.withdraw()		# hide the root tk window
	file_path = asksaveasfilename(defaultextension = '.xlsx',
									filetypes=[('Excel', '*.xlsx')],
									title = 'Save Template')
	root.destroy()
	export_text.value = basename(file_path)

export_template = Button(label='Export Template', button_type='success')
export_template.on_click(export_template_callback)
export_text = TextInput(value='<template file>', disabled=True)
export_row = row(export_template, export_text)

columns = [TableColumn(field='Sample File Name', title='Sample File Name', width=300),
			TableColumn(field='Marker', title='Marker', width=75),
			TableColumn(field='Allele', title='Allele', width=50),
			TableColumn(field='Area', title='Area', width=50),
			]

source = ColumnDataSource(data)
data_table = DataTable(columns=columns, source=source)

curdoc().add_root(column(results_row, host_row, donor_row, export_row, select_case))
curdoc().add_root(column(data_table, sizing_mode='stretch_height'))
curdoc().title = 'PTE'


''' Outline of functions
	get_host_file
		callback -> convert_fsa_to_csv
					make_dataframe
					apply_local_southern
					reindex_dataframe
					plot_graph
	get_donor_file
		callback ->	convert_fsa_to_csv
					make_dataframe
					apply_local_southern
					reindex_dataframe
					plot_graph
'''