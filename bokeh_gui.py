from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models.widgets import FileInput
from bokeh.models import TextInput, Button
import easygui
from convert_fsa_to_csv import convert_file, convert_file_content
from os.path import basename
import base64

def select_host_callback():
	fpath = convert_file()
	host_text.value = basename(fpath)

select_host = Button(label='Select Host', button_type='success')
select_host.on_click(select_host_callback)
host_text = TextInput(value='Select Host')
host_row = row(select_host, host_text)

def select_donor_callback():
	fpath = convert_file()
	donor_text.value = basename(fpath)

select_donor = Button(label='Select Donor', button_type='success')
select_donor.on_click(select_donor_callback)
donor_text = TextInput(value='Select Donor')
donor_row = row(select_donor, donor_text)

def select_results_callback():
	easygui.fileopenbox()

# select_results = FileInput(accept='.csv, .txt, .tsv')
select_results = Button(label='Select GeneMapper Results', button_type='success')
select_results.on_click(select_results_callback)
results_text = TextInput(value='Select GeneMapper Results')
results_row = row(select_results, results_text)

def export_template_callback():
	fname = easygui.filesavebox()
	# save_template_to_xlsx(fname)

export_template = Button(label='Export Template', button_type='success')
export_template.on_click(export_template_callback)

curdoc().add_root(column(host_row, donor_row, results_row, export_template))
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