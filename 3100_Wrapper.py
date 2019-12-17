#!/usr/bin/env python3

import pandas as pd
import subprocess
import argparse
import os
import sys


# Command-line argument parser.
def create_parser():
	parser = argparse.ArgumentParser(description="A simple wrapper for Convert_FSA_to_CSV_in_Batch.py and PeakFinder.py.")
	parser.add_argument("-d", dest="input", type=str, nargs=1 , help="Input FSA Directory.", required=True)
	args = parser.parse_args()
	return args


def main():

	myargs = create_parser()
	input_directory = myargs.input[0]
	input_directory = os.path.abspath(input_directory) + '/'

	# Note: You do not need shell=True to run a batch file or console-based executable.
	print('Converting FSA files...')
	converter = ['./converter/Convert_FSA_to_CSV_in_Batch.py', '-d', input_directory, '-co']
	subprocess.run(converter)

	print('Running Pick Peaks...')
	pick_peaks = ['./pick_peaks.py', input_directory]
	subprocess.run(pick_peaks)

if __name__ == '__main__':
	main()