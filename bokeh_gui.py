from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models.widgets import FileInput, DataTable, DateFormatter, TableColumn, RadioButtonGroup, RadioGroup, Select, TextAreaInput, MultiSelect, Div
from bokeh.models import TextInput, Button, ColumnDataSource, CDSView, IndexFilter, Panel, Tabs
from bokeh.server.server import Server

from convert_fsa_to_csv import convert_file, convert_file_content
from fsa import *
from extract_from_genemapper import *

from os.path import basename
import re
import pandas as pd
import copy
import openpyxl

import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfilename

from openpyxl.styles import Alignment


def bkapp(curdoc):
	pd.set_option('display.max_columns', 20)
	pd.set_option('display.width', 1000)
	pd.set_option('display.max_rows', 50)

	global_dict = {
		'results_files': [],
		'template_path': None,
		'p0_template_path': None,
		'p1_template_path': None,
		'p1_template_df': pd.DataFrame(),
		'p0_host': None,
		'p0_donors': [],
		'p0_current_case': None,
		'p1_current_case': None,
		'p1_host': None,
		'p1_donors':[],
	}

	def reduce_rows(df):
		# print(df)
		df = df.dropna(axis=1, how='all')
		cols = ['Sample File Name', 'Marker','Allele', 'Area']
		# print('********************************************')
		# print(df)
		df = df.dropna(axis=0, how='any', subset=cols)
		df = df.sort_values(by=cols, ascending=True)
		type_dict = {'Area':'int',
					'Size':'float',
					'Height':'int',
					'Data Point':'int'}
		for key, val in type_dict.items():
			if key in df.columns.tolist():
				df = df.astype({key:val})
		return df


	def p0c0_on_donor_change(attrname, old, new):
		# global p0c1_allele_table

		newest = [c for c in new if c not in old]
		if len(newest) > 0:
			global_dict['p0_current_case'] = newest[0]

			p0c1_table_title.text = str(global_dict['p0_current_case'])
			p0c1_allele_table.source.data = source_cases[newest[0]].data
			p0c1_allele_table.source.selected.indices = source_cases[newest[0]].selected.indices[:]
		refresh_p0_template_preview_table()


	def refresh_p0_template_preview_table():
			''' Preview Template '''
			wb = make_template_wb(host_case=p0c0_select_host_case.value, donor_cases=p0c0_select_donor_cases.value, df_cases=df_cases, host_name=p1c0_enter_host_name.value)
			ws = wb.worksheets[0]
			df = pd.DataFrame(ws.values)
			df.loc[-1] = ''
			# print('Inside of refresh_p0_template_preview_table')
			# print(df)
			# df.loc[-1] = df.columns.tolist()
			df.index = df.index + 1
			df.sort_index(inplace=True)
			col_letters = [openpyxl.utils.get_column_letter(int(i)+1) for i in df.columns.tolist()]
			df.columns = col_letters
			df = df.fillna('')
			df_col = df.columns.tolist()
			columns = [TableColumn(field=col, title=col, width=75) for col in df_col[0]]
			columns.extend([TableColumn(field=col, title=col, width=50) for col in df_col[1:-2]])
			columns.extend([TableColumn(field=col, title=col, width=250) for col in df_col[-2:]])
			p0c2_template_table.columns = columns
			p0c2_template_table.source.data = ColumnDataSource(df).data


	def p0c0_on_host_click(attrname, old, new):
		# global source_cases, p0c1_allele_table

		global_dict['p0_current_case'] = new
		p0c1_table_title.text = str(global_dict['p0_current_case'])
		p0c1_allele_table.source.data = source_cases[new].data
		p0c1_allele_table.source.selected.indices = source_cases[new].selected.indices[:]


	def pNc0_on_results_click():
		# global source_cases, df, p0c1_allele_table, df_cases

		root = tk.Tk()
		root.attributes("-topmost", True)
		root.withdraw()		# hide the root tk window
		file_path = askopenfilename(filetypes = (('Text file', '*.txt'),
												('Comma Separated Values','*.csv'),
												('Tab Separated Values','*.tsv'),
											),
									title = 'Choose GeneMapper results file.',
									initialdir=r'X:\Hospital\Genetics Lab\DNA_Lab\3-Oncology Tests\Engraftment\ABI PTE Runs'
								)
		root.destroy()
		if basename(file_path) is not None and basename(file_path) != '' and basename(file_path) not in global_dict['results_files']:
			global_dict['results_files'].append(basename(file_path))
			pNc0_results_text.value = '\n'.join(global_dict['results_files'])

			df_new = use_csv_module(file_path)
			if not df_new.empty:
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

				p0c0_select_host_case.options = list(source_cases.keys())
				p0c0_select_donor_cases.options = list(source_cases.keys())
				p1c0_select_samples.options = list(source_cases.keys())

		if len(global_dict['results_files']) == 1:
			p0c0_select_host_case.value = list(source_cases.keys())[0]


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


	def p0c0_on_export_template_click():
		root = tk.Tk()
		root.attributes("-topmost", True)
		root.withdraw()		# hide the root tk window
		file_path = asksaveasfilename(defaultextension = '.xlsx',
										filetypes=[('Excel', '*.xlsx')],
										title = 'Save Template',
										initialfile=p0c0_enter_host_name.value)
		root.destroy()
		if file_path.endswith('.xlsx'):
			make_template_wb(file_path=file_path,
								host_case=p0c0_select_host_case.value,
								donor_cases=p0c0_select_donor_cases.value,
								df_cases=df_cases,
								host_name=p1c0_enter_host_name.value)

	def p0c1_on_select_alleles_change(attrname, old, new):
		# global df_cases

		indices = p0c1_allele_table.source.selected.indices[:]
		source_cases[global_dict['p0_current_case']].selected.indices = indices[:]
		df_cases[global_dict['p0_current_case']].loc[:,'Selected'] = False
		df_cases[global_dict['p0_current_case']].loc[indices,'Selected'] = True
		refresh_p0_template_preview_table()


	def p1c0_on_select_template_click():
		root = tk.Tk()
		root.attributes("-topmost", True)
		root.withdraw()		# hide the root tk window
		file_path = askopenfilename(filetypes=[
												('Excel', '*.xlsx'),
												('Excel', '*.xls'),
											],
									title = 'Choose Template',
									initialdir=r'X:\Hospital\Genetics Lab\DNA_Lab\3-Oncology Tests\Engraftment\Allele Charts')
		root.destroy()

		# print('file_path = {}'.format(file_path))

		''' Convert xls to xlsx if needed '''
		if file_path.endswith('.xls'):
			file_path = convert_xls_to_xlsx(file_path)
		if file_path is not None and file_path != '':
		# drop_empty_columns_and_adjust_formulae(file_path)
			''' Update template_text box'''
			p1c0_template_text.value = basename(file_path)
			global_dict['template_path'] = file_path
			# global template_path
			# template_path = file_path
			# df = fill_template_with_areas_2(template=global_dict['template_path'])
			# makeshift_results_dictionary(template=global_dict['template_path'])
			wb = make_template_from_existing_template(template=global_dict['template_path'])
			patient_name = get_patient_name_from_header(global_dict['template_path'])
			# print('patient_name = {}'.format(patient_name))
			if patient_name is None or patient_name == '':
				template_path = global_dict['template_path']
				patient_name = basename(template_path)
				patient_name = patient_name.replace('.xlsx', '')
				patient_name = patient_name.replace('.xls', '')
			p1c0_enter_host_name.value = patient_name
			df = pd.DataFrame(wb.worksheets[0].values)
			global_dict['p1_template_df'] = df.copy(deep=True)
			'''	But also fill the template if samples are already selected! '''
			# if global_dict['p1_current_case'] is not None:
				# print('\t\t***** NOW WE GONNA REFRESH THIS TABLE \'CAUSE YOLO')
			refresh_p1_template_preview_table()

	def check_if_multiple_donors():
		pass


	def p1c0_on_export_results_click():
		# global template_path
		template_path = global_dict['template_path']

		# header = get_header(template_path)
		# patient_name = header.center.text
		patient_name = p1c0_enter_host_name.value

		for sample in p1c0_select_samples.value:
			'''	Construct output file name '''
			output_file_name = patient_name + ' ' + re.sub(r'_PTE.*$', '', sample)
			root = tk.Tk()
			root.attributes("-topmost", True)
			root.withdraw()		# hide the root tk window
			output_file_name = asksaveasfilename(defaultextension='.xlsx',
											filetypes=[('Excel', '*.xlsx')],
											title='Populate results for {}'.format(sample),
											initialfile = output_file_name)
			root.destroy()

			if output_file_name is not None and output_file_name != '':
				if output_file_name.endswith('.xlsx'):
					pass
				elif output_file_name.endswith('.xls'):
					output_file_name = output_file_name + 'x'
				else:
					output_file_name = output_file_name + '.xlsx'

				df = df_cases[sample]
				# print(df)
				results = build_results_dict(df)
				# print(results)
				df_filled = fill_template_with_areas(res=results, sample_name=sample, template=global_dict['p1_template_df'])
				# global_dict['p1_template_df'] = df_filled.copy(deep=True)
				# print(df_filled)

				# col_letters = [openpyxl.utils.get_column_letter(int(i)+1) for i in df_filled.columns.tolist()]
				# df_filled.columns = col_letters
				df_filled = df_filled.fillna('')
				df_filled.to_excel(output_file_name, index=False, header=False)
				fix_formatting(filename=output_file_name, patient_name=p1c0_enter_host_name.value)


	def p1c0_on_select_samples_change(attrname, old, new):

		template_path = global_dict['template_path']

		newest = [c for c in new if c not in old]
		if len(newest) > 0:
			'''	Update allele table '''
			global_dict['p1_current_case'] = newest[0]
			refresh_p1_allele_table()
			refresh_p1_template_preview_table()

	def refresh_p1_allele_table():
		if len(str(global_dict['p1_current_case'])) > 0:
			p1c1_table_title.text = str(global_dict['p1_current_case'])
		else:
			p1c1_table_title.text = '&lt;sample&gt;'
		source = source_cases.get('p1_current_case',ColumnDataSource())
		p1c1_allele_table.source.data = source_cases[global_dict['p1_current_case']].data
		# p1c1_allele_table.source.data = source.data
		p1c1_allele_table.source.selected.indices = source_cases[global_dict['p1_current_case']].selected.indices[:]
		# p1c1_allele_table.source.selected.indices = source.selected.indices[:]


	def refresh_p1_template_preview_table():

		df = df_cases.get(global_dict['p1_current_case'],pd.DataFrame())
		# print("df_cases[global_dict['p1_current_case']]")
		# print(df)
		results = build_results_dict(df)
		# pprint(results)
		df_filled = fill_template_with_areas(res=results,
									sample_name=global_dict['p1_current_case'],
									template=global_dict['p1_template_df'])
		global_dict['p1_template_df'] = df_filled.copy(deep=True)
		# print('df_filled')
		# print(df_filled)
		if not df_filled.empty:
			# print(df_filled)
			df_filled.loc[-1] = ''
			# print(df_filled)
			# print(df_filled.index.tolist())
			df_filled.index = df_filled.index + 1
			df_filled.sort_index(inplace=True)

			col_letters = [openpyxl.utils.get_column_letter(int(i)+1) for i in df_filled.columns.tolist()]
			df_filled.columns = col_letters
			df_filled = df_filled.fillna('')
			df_col = df_filled.columns.tolist()
			columns = [TableColumn(field=col, title=col, width=75) for col in df_col[0]]
			columns.extend([TableColumn(field=col, title=col, width=50) for col in df_col[1:-2]])
			columns.extend([TableColumn(field=col, title=col, width=250) for col in df_col[-2:]])
			p1c2_template_table.columns = columns
			p1c2_template_table.source.data = ColumnDataSource(df_filled).data


	def p1c1_on_allele_table_edit(attrname, old, new):
		# print(p1c1_allele_table.source.data)
		pass


	# results_files = []
	source_cases = {}
	df_cases = {}

	pNc0_select_results = Button(label='Add GeneMapper Results', button_type='success')
	pNc0_select_results.on_click(pNc0_on_results_click)
	pNc0_results_text = TextAreaInput(value='<results file>', disabled=True, rows=5)


	# select_host_case = Select(title='Select Host', options=list(source_cases.keys()))
	p0c0_select_host_case = Select(title='Select Host', options=[])
	p0c0_select_host_case.on_change('value', p0c0_on_host_click)


	p0c0_select_donor_cases = MultiSelect(title='Select Donor(s) <ctrl+click to multiselect>',
									options=[],
									size=20)
	p0c0_select_donor_cases.on_change('value', p0c0_on_donor_change)


	p0c0_export_template = Button(label='Export Template', button_type='warning')
	p0c0_export_template.on_click(p0c0_on_export_template_click)


	p0c0_enter_host_name = TextInput(title='Enter Host Name', value='<type host name here>')
	p1c0_enter_host_name = TextInput(title='Enter Host Name', value='<type host name here>')


	p1c0_select_template = Button(label='Select Template', button_type='success')
	p1c0_select_template.on_click(p1c0_on_select_template_click)
	p1c0_template_text = TextAreaInput(value='<template file>', disabled=True)


	p1c0_select_samples = MultiSelect(title='Select Sample(s) <ctrl+click to multiselect>',
										options=[],
										size=20)
	p1c0_select_samples.on_change('value', p1c0_on_select_samples_change)

	p1c0_export_results = Button(label='Export Results To Excel File', button_type='warning')
	p1c0_export_results.on_click(p1c0_on_export_results_click)

	p1c1_table_title = Div(text='&lt;sample&gt;', sizing_mode='fixed')


	p0c1_table_title = Div(text='&lt;sample&gt;', sizing_mode='fixed')
	p0c2_table_title = Div(text='Template', sizing_mode='fixed')
	p1c2_table_title = Div(text='Template', sizing_mode='fixed')

	# p1c2_table_title = Div(text='&lt;sample&gt;', sizing_mode='fixed')


	columns = [#TableColumn(field='Sample File Name', title='Sample File Name', width=300),
				TableColumn(field='Marker', title='Marker', width=75),
				TableColumn(field='Allele', title='Allele', width=50),
				TableColumn(field='Area', title='Area', width=50)]


	source = ColumnDataSource()
	source.selected.on_change('indices', p0c1_on_select_alleles_change)

	p0c1_allele_table = DataTable(columns=columns, source=source, selectable='checkbox', fit_columns=True, sizing_mode='stretch_height', width=300)
	p0c2_template_table = DataTable(source=ColumnDataSource(), fit_columns=True, sizing_mode='stretch_both')
	p1c2_template_table = DataTable(source=ColumnDataSource(), fit_columns=True, editable=True)

	p1c1_allele_table = DataTable(columns=columns, source=ColumnDataSource(), selectable='checkbox', fit_columns=True, sizing_mode='stretch_height', width=300, editable=True)
	p1c1_allele_table.source.on_change('data', p1c1_on_allele_table_edit)

	p0c0 = column(p0c0_enter_host_name,
					pNc0_select_results,
					pNc0_results_text,
					p0c0_select_host_case,
					p0c0_select_donor_cases,
					p0c0_export_template,
					sizing_mode='fixed')

	p0c1 = column(p0c1_table_title, p0c1_allele_table, sizing_mode='stretch_height')

	p0c2 = column(p0c2_table_title, p0c2_template_table, sizing_mode='stretch_both')
	# p0c2 = column(p0c2_template_table)

	p1c0 = column(p1c0_enter_host_name,
					pNc0_select_results,
					pNc0_results_text,
					p1c0_select_template,
					p1c0_template_text,
					p1c0_select_samples,
					p1c0_export_results)

	p1c1 = column(p1c1_table_title, p1c1_allele_table, sizing_mode='stretch_height')

	p1c2 = column(p1c2_table_title, p1c2_template_table, sizing_mode='scale_height')

	child_0 = row(p0c0, p0c1, p0c2, sizing_mode='stretch_both')
	# child_0 = row(p0c0, p0c1, p0c2)
	child_1 = row(p1c0, p1c1, p1c2, sizing_mode='stretch_height')
	tab1 = Panel(child=child_0, title='Make Template')
	tab2 = Panel(child=child_1, title='Populate Results')
	tabs = Tabs(tabs=[tab1, tab2])
	curdoc.add_root(tabs)
	curdoc.title = 'PTE'


# Setting num_procs here means we can't touch the IOLoop before now, we must
# let Server handle that. If you need to explicitly handle IOLoops then you
# will need to use the lower level BaseServer class.
server = Server({'/': bkapp}, num_procs=1)
server.start()

if __name__ == '__main__':
	print('Opening Bokeh application on http://localhost:5006/')

	server.io_loop.add_callback(server.show, "/")
	server.io_loop.start()