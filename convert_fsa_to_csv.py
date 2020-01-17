#!/usr/bin/env python3

from Bio import SeqIO
from collections import defaultdict
import pandas as pd
import argparse
import os
import sys
# import easygui
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, askdirectory



# Command-line argument parser.
def create_parser():
	parser = argparse.ArgumentParser(description="This program converts Thermo Fisher 3100 Genetic Analyzer FSA files to CSV.")
	parser.add_argument("-d", dest="input", type=str, nargs=1 , help="Input FSA Directory.", required=True)
	parser.add_argument("-co", dest="channels_only", action='store_true', default=False, help="Option returns only Channels in CSV.", required=False)
	#parser.add_argument("-o", dest="output", type=str, nargs=1, help="output CSV file", required=True) 
	args = parser.parse_args()
	return args

def create_dataframe(record, keys):
	'''
	Description: Returns a dataframe containing all the data values for FSA/ABI file for 3130 Sequencer.
	url: https://projects.nfstc.org/workshops/resources/articles/ABIF_File_Format.pdf
	'''
	DATA_list = ['DATA1','DATA2','DATA3','DATA4','DATA105']
	# cols = [record.name + '.fsa.channel_1', record.name + '.fsa.channel_2', record.name + '.fsa.channel_3', record.name + '.fsa.channel_4']
	channels = [record.annotations['abif_raw'][key] for key in keys if key in DATA_list]
	cols = [record.name + '.fsa.channel_' + str(i+1) for i in range(len(channels))]
	df = pd.DataFrame(channels)
	df = df.T
	df.columns = cols
	return df

def metadata_dataframe(record, keys):
	'''
	Description: Returns the a dataframe containing metadata associated to the FSA/ABI file for 3130 Sequencer.
	url: https://projects.nfstc.org/workshops/resources/articles/ABIF_File_Format.pdf
	'''
	ABI_3130 = {
	'APFN1': 'Sequence Analysis parameters file name',
	'APrN1': 'Analysis Protocol settings name',
	'APrV1': 'Analysis Protocol settings version',
	'APrX1': 'Analysis Protocol XML string',
	'APXV1': 'Analysis Protocol XML schema version',
	'CMNT1': 'Comment About Sample (Optional)',
	'CpEP1': 'Is Cappillary Machine?',
	'CTID1': 'Plate Barcode',
	'CTNM1' : 'Container Name',
	'CTOw1': 'Container owner',
	'CTTL1': 'Comment Title',
	'DSam1': 'Downsampling Factor',
	'Dye#1': 'Number of Dyes',
	'DyeB1': 'Dye 1 Significance S for standard, space for sample',
	'DyeB2': 'Dye 2 Significance S for standard, space for sample',
	'DyeB3': 'Dye 3 Significance S for standard, space for sample',
	'DyeB4': 'Dye 4 Significance S for standard, space for sample',
	'DyeB5': 'Dye 5 Significance S for standard, space for sample',
	'DyeN1': 'Dye 1 Name',
	'DyeN2': 'Dye 2 Name',
	'DyeN3': 'Dye 3 Name',
	'DyeN4': 'Dye 4 Name',
	'DyeW1': 'Dye 1 Wavelength',
	'DyeW2': 'Dye 2 Wavelength',
	'DyeW3': 'Dye 3 Wavelength',
	'DyeW4': 'Dye 4 Wavelength',
	'DySN1': 'Dye Set Name',
	'EPVt1': 'Electrophoresis Voltage settings',
	'EVNT1': 'Start run event',
	'EVNT2': 'Stop run event',
	'EVNT3': 'Start collection event',
	'EVNT4': 'Stop collection event',
	'FWO_1': 'Base order',
	'GTyp1': 'Get type description',
	'HCFG1': 'Instrument Class',
	'HCFG2': 'Instrument family', 
	'HCFG3': 'Official Instrument Name',
	'HCFG4': 'Instrument parameters',
	'InSc1': 'Injection Time (seconds)',
	'InVt1': 'Injection Voltage (Volts)',
	'LANE1': 'Lane/Cappillary',
	'LIMS1': 'Sample Tracking ID',
	'LNTD1': 'Length to Detector',
	'LsrP1': 'Laser Power Setting (microWatts)',
	'MCHN1': 'Instrument Name and Serial Number',
	'MODF1': 'Data collection module file',
	'MODL1': 'Model number',
	'NAVG1': 'Pixels average per lane',
	'NLNE1': 'Number of cappilaries',
	'OfSc1': 'List of scans that are marked off scale in collection (Optional)',
	'Ovrl1': 'One value for each dye. List of scan number indices for scans with colo data values > 32767. Values cannot be greater than 32000. (Optional)',
	'OvrV1': 'One value for each dye. List of color data values for the locations liusted in the Ovrl tag. Number of OvrV tags must be equal to the number of Ovrl tags (Optional)',
	'PDMF1': 'Mobility file 1',
	'PDMF2': 'Mobility file 2',
	'PRJT1': 'SeqScape project templat name',
	'PROJ1': 'SeqScape project name',
	'PXLB1': 'Pixel bin size',
	'Rate1': 'Scanning rate',
	'RGCm1': 'Results group comment (optional)',
	'RGNm1': 'Results group name',
	'RGOw1': 'The name entered as the owner of a results group, in  the Results Group editor (optional)',
	'RMdN1': 'Run module name (same as MODF1)',
	'RMdV1': 'Run module version',
	'RMdX1': 'Run module XML string',
	'RMXV1': 'Run module XML schema version',
	'RPrN1': 'Run Protocol name',
	'RPrV1': 'Run Protocol version',
	'RUND1': 'Run Start Date',
	'RUND2': 'Run Stop Date',
	'RUND3': 'Date collection Start Date',
	'RUND4': 'Date collection Stop Date',
	'RunN1': 'Run name',
	'RUNT1': 'Run Start Time',
	'RUNT2': 'Run Stop Time',
	'RUNT3': 'Date Collection Start Time',
	'RUNT4': 'Date Collection Stop Time',
	'Satd1': 'Array of longs representing the scan numbers of data points, which are flagged as saturated by data collection (optional)',
	'Scal1': 'Rescaling divisor of color data',
	'Scan1': 'Number of scans (legacy - use SCAN)',
	'SCAN1': 'Number of scans',
	'SMED1': 'Polymer expiration date',
	'SMLt1': 'Polymer number lot',
	'SMPL1': 'Sample Name',
	'SPEC1': 'SeqScape specimen name',
	'SVER1': 'Data collection software version',
	'SVER3': 'Data collection firmware version',
	'Tmpr1': 'Run Temperature Setting (degrees C)',
	'TUBE1': 'Well ID',
	'User1': 'Name of user who created plate',
	}

	metadata = []
	for key, description in ABI_3130.items():
		if 'DATA' not in key:
			if key in record.annotations['abif_raw'].keys():
				metadata.append([str(key), str(description), record.annotations['abif_raw'][key]])

	df = pd.DataFrame(metadata)
	df = df.T
	cols = df.iloc[0:1,].values[0].tolist()
	df = df.iloc[1:len(df),]
	df.columns = cols
	return df.reset_index()

def find_3130_files(dir_path):
	files = [os.path.join(root, file) for root, dirs, files in os.walk(dir_path) for file in files if file.endswith('.fsa')]
	return files

def convert_file_content(file_content, channels_only=True):
	record = SeqIO.read(file_content, 'abi')
	keys = record.annotations['abif_raw'].keys()

	data = create_dataframe(record, keys)
	metadata = metadata_dataframe(record, keys)

	results = data

	if not channels_only:
		metadata = metadata_dataframe(record, keys)
		results = pd.concat([data, metadata], axis=1)
		del results['index']
	return results

def convert_file(file_path=None, channels_only=True):
	root = tk.Tk()
	root.withdraw()		# hide the root tk window

	if file_path == None:
		file_path = askopenfilename(filetypes = (('Fragment Size Analysis', '*.fsa'),
											('Comma Separated Values','*.csv'),
										),
								title = 'Choose an FSA file.'
								)
	root.destroy()		# kill the root tk window

	abs_input_file = os.path.abspath(file_path)
	outfile_path = abs_input_file.replace('.fsa', '.csv')

	if not os.path.isfile(outfile_path):
		record = SeqIO.read(abs_input_file, 'abi')
		keys = record.annotations['abif_raw'].keys()

		data = create_dataframe(record, keys)
		metadata = metadata_dataframe(record, keys)

		results = data

		if not channels_only:
			metadata = metadata_dataframe(record, keys)
			results = pd.concat([data, metadata], axis=1)
			del results['index']
		results.to_csv(outfile_path, index=False, header=True)
	return outfile_path

def convert_folder(dir_path=None, channels_only=True):
	root = tk.Tk()
	root.withdraw()		# hide the root tk window

	if dir_path == None:
		# dir_path = easygui.diropenbox()
		dir_path = askdirectory(filetypes = (('Fragment Size Analysis', '*.fsa'),
											('Comma Separated Values','*.csv'),
										),
								title = 'Choose folder of FSA files.'
								)
	root.destroy()		# kill the root tk window

	abs_path_dir = os.path.abspath(dir_path)
	files = find_3130_files(abs_path_dir)
	outfile_paths = []

	print('Found {} fsa files. Beginning conversion to csv'.format(len(files)))

	for input_file in files:
		abs_input_file = os.path.abspath(input_file)
		outfile_path = abs_input_file.replace('.fsa', '.csv')
		outfile_paths.append(outfile_path)
		if not os.path.isfile(outfile_path):
			record = SeqIO.read(abs_input_file, 'abi')
			keys = record.annotations['abif_raw'].keys()

			data = create_dataframe(record, keys)
			metadata = metadata_dataframe(record, keys)

			results = data

			if not channels_only:
				metadata = metadata_dataframe(record, keys)
				results = pd.concat([data, metadata], axis=1)
				del results['index']
			results.to_csv(outfile_path, index=False, header=True)
	return outfile_paths

def main():
	myargs = create_parser()
	input_dir = myargs.input[0]

	abs_path_dir = os.path.abspath(input_dir)
	files = find_3130_files(abs_path_dir)

	print('Found {} fsa files. Beginning conversion to csv'.format(len(files)))

	for input_file in files:
		abs_input_file = os.path.abspath(input_file)
		outfile_path = abs_input_file.split('.')[0] + '.csv'

		record = SeqIO.read(abs_input_file, 'abi')
		keys = record.annotations['abif_raw'].keys()

		data = create_dataframe(record, keys)
		metadata = metadata_dataframe(record, keys)

		results = pd.concat([data, metadata], axis=1)
		del results['index']

		if myargs.channels_only:
			results = results.iloc[:,0:4]
			results.to_csv(outfile_path, index=False, header=True)
		else:
			results.to_csv(outfile_path, index=False, header=True)


if __name__ == '__main__':
	main()
