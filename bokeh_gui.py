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

results_files = []
cases = []

def select_results_callback():
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

	df = use_csv_module(file_path)
	df = reduce_rows(df)
	source = ColumnDataSource(df)
	columns = [TableColumn(field='Sample File Name', title='Sample File Name', width=300),
				TableColumn(field='Marker', title='Marker', width=75),
				TableColumn(field='Allele', title='Allele', width=50),
				TableColumn(field='Area', title='Area', width=50),
				]
	data_table = DataTable(source=source, columns=columns)

	cases = sorted(list(set(cases) | set(df['Sample File Name'].to_list())))
	# radio_button_group = RadioGroup(labels=cases)
	select = Select(options=cases)

	curdoc().add_root(select)
	curdoc().add_root(column(data_table, sizing_mode='stretch_height'))
	# curdoc().add_root(column(data_table))
	# curdoc().add_root(data_table)

# select_results = FileInput(accept='.csv, .txt, .tsv')
# df = pd.DataFrame()
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

curdoc().add_root(column(results_row, host_row, donor_row, export_row))
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