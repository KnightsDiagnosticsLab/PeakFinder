# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['extract_from_genemapper.py'],
             pathex=['C:\\Users\\ghanih\\Documents\\GitHub\\PeakFinder'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['matplotlib', 'scipy', 'biopython', 'bokeh', 'openslide-python', 'PeakUtils', 'outlier-utils', 'Pillow', 'pylint', 'python-dateutil', 'Jinja2', 'pywebview', 'tornado', 'setuptools', 'py2exe', 'pip', 'MarkupSafe'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='extract_from_genemapper',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
