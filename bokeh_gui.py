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