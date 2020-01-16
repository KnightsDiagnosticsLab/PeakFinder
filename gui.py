#!/usr/bin/env python3
# import PTE as pte
from convert_fsa_to_csv import convert_folder
import tkinter as tk
from tkinter import ttk
import os

def main():
    ''' tkinter based gui. Has crappy support for interactive plots.
    '''
    window = tk.Tk()
    window.title('PTE')
    tab_control = ttk.Notebook(window)
    tab1 = ttk.Frame(tab_control)
    tab2 = ttk.Frame(tab_control)
    tab_control.add(tab1, text='Build New Profile')
    tab_control.add(tab2, text='Interpret Results')
    tab_control.pack(expand=1, fill='both')

    host = tk.Label(tab1, text='No host file selected')
    host.grid(column=1, row=0, sticky=tk.W)
    def get_fsa():
        host_fpath = tk.filedialog.askopenfilename()
        host_fname = os.path.basename(host_fpath)
        host.configure(text=host_fname or 'No host file selected')
    host_btn = tk.Button(tab1, text='Select Host', command=get_fsa)
    host_btn.grid(column=0, row=0, sticky=tk.E)

    # donors = tk.Message(tab1, text='No donor files selected')
    donors = tk.Label(tab1, text='No donor files selected')
    donors.grid(column=1, row=1, sticky=tk.W)
    def get_donors():
        donor_fpaths = tk.filedialog.askopenfilenames()
        donor_fnames = '\n'.join([os.path.basename(f) for f in donor_fpaths])
        donors.configure(text=donor_fnames or 'No donor files selected')
    donors_btn = tk.Button(tab1, text='Select Donor(s)', command=get_donors)
    donors_btn.grid(column=0, row=1, sticky=tk.NE)

    window.mainloop()

if __name__ == '__main__':
    main()
