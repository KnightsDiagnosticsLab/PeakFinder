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
from fsa import use_csv_module, fix_formatting
import copy
import openpyxl
from openpyxl.styles import Alignment

pd.set_option('display.max_columns', 20)


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

	newest = [c for c in new if c not in old]
	if len(newest) > 0:
		current_case = newest[0]
		data_table.source.data = source_cases[newest[0]].data
		data_table.source.selected.indices = source_cases[newest[0]].selected.indices[:]


def on_host_click(attrname, old, new):
	global source_cases, data_table, current_case

	current_case = new
	data_table.source.data = source_cases[new].data
	data_table.source.selected.indices = source_cases[new].selected.indices[:]


def on_results_click():
	global source_cases, results_files, df, source, data_table, df_cases

	root = tk.Tk()
	root.attributes("-topmost", True)
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
		select_donor_cases.options = list(source_cases.keys())
		select_samples.options = list(source_cases.keys())

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


def make_template_xlsx(file_path):

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
	host_case = select_host_case.value
	df_host = df_cases[host_case]
	df_host = df_host.loc[df_host['Selected'] == True]

	donor_cases = select_donor_cases.value
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

			percent_host.value = '=100*SUM({}:{})/SUM({}:{})'.format(h1_area.coordinate,
																h2_area.coordinate,
																d1_area.coordinate,
																h2_area.coordinate)

		if len(h) == 1 and len(h | d) == 3:
			f1.value = '{}'.format(h1_alle_val)
			f1.alignment = Alignment(horizontal='center')

			f2.value = '{} + {} + {}'.format(h1_alle_val,
											d1_alle_val,
											d2_alle_val)

			percent_host.value = '=100*{}/SUM({}:{})'.format(h1_area.coordinate,
														d1_area.coordinate,
														h1_area.coordinate)

		if len(d) == 1 and len( h | d) == 3:
			f1.value = '{} + {}'.format(h1_alle_val,
										h2_alle_val)
			f1.alignment = Alignment(horizontal='center')

			f2.value = '{} + {} + {}'.format(h1_alle_val,
											h2_alle_val,
											d1_alle_val)

			percent_host.value = '=100*SUM({}:{})/SUM({},{}:{})'.format(h1_area.coordinate,
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

			f2.value = '(1-(2*{}/({}+{})))'.format(A_alle_val,
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

			f1.value = '2 * {}'.format(A_alle_val)
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

	'''	Save the file before running fix_formatting
	'''
	wb.save(file_path)

	'''	Fix formatting
	'''
	fix_formatting(file_path)
	print('Done saving {}'.format(file_path))


def on_export_template_click():
	root = tk.Tk()
	root.attributes("-topmost", True)
	root.withdraw()		# hide the root tk window
	file_path = asksaveasfilename(defaultextension = '.xlsx',
									filetypes=[('Excel', '*.xlsx')],
									title = 'Save Template')
	root.destroy()
	# print('******** file_path = {} '.format(file_path))
	if file_path.endswith('.xlsx'):
		make_template_xlsx(file_path)


def on_selected_change(attrname, old, new):
	global current_case, df_cases

	indices = data_table.source.selected.indices[:]
	source_cases[current_case].selected.indices = indices[:]
	df_cases[current_case].loc[:,'Selected'] = False
	df_cases[current_case].loc[indices,'Selected'] = True


def on_select_template_click():
	root = tk.Tk()
	root.attributes("-topmost", True)
	root.withdraw()		# hide the root tk window
	file_path = askopenfilename(filetypes=[('Excel', '*.xlsx')],
								title = 'Choose Template')
	root.destroy()

	''' Update template_text box'''
	template_text.value = basename(file_path)
	global template_path
	template_path = file_path
	wb = openpyxl.load_workbook(file_path)
	ws = wb.worksheets[0]
	# df = pd.read_excel(template_path)
	df = pd.DataFrame(ws.values)
	# print(df)
	df.loc[-1] = ''
	# df.loc[-1] = df.columns.tolist()
	df.index = df.index + 1
	df.sort_index(inplace=True)
	col_letters = [openpyxl.utils.get_column_letter(i+1) for i in df.columns.tolist()]
	df.columns = col_letters
	df = df.fillna('')
	columns = [TableColumn(field=col, title=col) for col in df.columns.tolist()]
	template_table.columns = columns
	template_table.source.data = ColumnDataSource(df).data


def populate_template_xlsx(file_path):
	# assert file_path.endswith('.xlsx')
	# wb = openpyxl.load_workbook(file_path)
	# ws = wb.worksheets[0]
	pass


def on_populate_results_click():
	for sample in select_samples.value:
		root = tk.Tk()
		root.attributes("-topmost", True)
		root.withdraw()		# hide the root tk window
		file_path = asksaveasfilename(defaultextension='.xlsx',
										filetypes=[('Excel', '*.xlsx')],
										title='Populate results for {}'.format(sample))
		root.destroy()

		populate_template_xlsx(file_path)

results_files = []
source_cases = {}
df_cases = {}

select_results = Button(label='Add GeneMapper Results', button_type='success')
select_results.on_click(on_results_click)
results_text = TextAreaInput(value='<results file>', disabled=True, rows=5)


# select_host_case = Select(title='Select Host', options=list(source_cases.keys()))
select_host_case = Select(title='Select Host', options=[])
select_host_case.on_change('value', on_host_click)


select_donor_cases = MultiSelect(title='Select Donor(s) <ctrl+click to multiselect>',
								options=[],
								size=20)
select_donor_cases.on_change('value', on_donor_change)


export_template = Button(label='Export Template', button_type='warning')
export_template.on_click(on_export_template_click)


enter_host_name = TextInput(title='Enter Host Name', value='<type host name here>')


select_template = Button(label='Select Template', button_type='success')
select_template.on_click(on_select_template_click)
template_text = TextAreaInput(value='<template file>', disabled=True)


select_samples = MultiSelect(title='Select Sample(s) <ctrl+click to multiselect>',
									options=[],
									size=20)


populate_results = Button(label='Populate Results', button_type='warning')
populate_results.on_click(on_populate_results_click)

columns = [TableColumn(field='Sample File Name', title='Sample File Name', width=300),
			TableColumn(field='Marker', title='Marker', width=75),
			TableColumn(field='Allele', title='Allele', width=50),
			TableColumn(field='Area', title='Area', width=50)]


source = ColumnDataSource()
source.selected.on_change('indices', on_selected_change)
data_table = DataTable(columns=columns, source=source, selectable='checkbox')
template_table = DataTable(source=ColumnDataSource())


col_0 = column(enter_host_name,
				select_results,
				results_text,
				select_host_case,
				select_donor_cases,
				export_template)

col_1 = column(data_table, sizing_mode='stretch_height')

col_2 = column(select_results,
				results_text,
				select_template,
				template_text,
				select_samples,
				populate_results)

col_3 = column(template_table, sizing_mode='stretch_both')

child_0 = row(col_0, col_1, sizing_mode='stretch_height')
child_1 = row(col_2, col_3, sizing_mode='stretch_height')
tab1 = Panel(child=child_0, title='Make Template')
tab2 = Panel(child=child_1, title='Populate Results')
tabs = Tabs(tabs=[tab1, tab2])
curdoc().add_root(tabs)
curdoc().title = 'PTE'


''' Outline of functions
	Input box for Host's name
	on_case_change
		DONE: add logic of pre-selecting certain rows
		color code rows by Marker -> harder than it should be!
		show live export template
		
'''