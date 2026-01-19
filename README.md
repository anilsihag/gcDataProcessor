# GC Data Processor GUI (PyQt6)

A PyQt6 + Matplotlib GUI for GC chromatogram processing:
- Load multiple CSV/TXT files
- Peak detection + integration
- Add/delete peaks
- Drag integration bounds
- Assign gas labels + calibration (Excel)
- Export results to Excel

## Requirements
Python 3.10+ recommended

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

##
'''bash
pyhton -m gcdataprocessor
