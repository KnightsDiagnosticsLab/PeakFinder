from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models.widgets import FileInput
from bokeh.models import TextInput



def update_textbox(attrname, old, new):
	text.value = new

file_input = FileInput(accept='.fsa, .csv')
file_input.on_change('filename', update_textbox)

text = TextInput(title="title", value='my sine wave')

curdoc().add_root(row(file_input, text))
curdoc().title = "Sliders"


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