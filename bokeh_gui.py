from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models.widgets import FileInput
from bokeh.models import TextInput, Button
import easygui

def select_host_callback(attrname, old, new):
	host_text.value = new

select_host = FileInput(accept='.fsa')
select_host.on_change('filename', select_host_callback)
host_text = TextInput(title="Host", value='Select Host')
host_row = row(select_host, host_text)

def select_donor_callback(attrname, old, new):
	donor_text.value = new

select_donor = FileInput(accept='.fsa')
select_donor.on_change('filename', select_donor_callback)
donor_text = TextInput(title="Donor", value='Select Donor')
donor_row = row(select_donor, donor_text)

def select_results_callback(attrname, old, new):
	results_text.value = new

select_results = FileInput(accept='.csv, .txt, .tsv')
select_results.on_change('filename', select_results_callback)
results_text = TextInput(title='GeneMapper Results', value='Select GeneMapper Results')
results_row = row(select_results, results_text)

def export_template_callback():
	fname = easygui.filesavebox(filetypes='.xlsx')
	# save_template_to_xlsx(fname)

export_template = Button(label='Export Template', button_type='success')
export_template.on_click(export_template_callback)
export_row = row(export_template)

curdoc().add_root(column(host_row, donor_row, export_row))
curdoc().title = "PTE"


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