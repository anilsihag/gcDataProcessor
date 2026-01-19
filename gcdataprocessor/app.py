import sys
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import trapezoid
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                             QListWidget, QSplitter, QMessageBox, QCheckBox, 
                             QTableWidget, QTableWidgetItem, QHeaderView, 
                             QFrame, QGridLayout, QDoubleSpinBox, QComboBox)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# ==================================================
# Core Math Logic
# ==================================================

def find_valleys(signal):
    return find_peaks(-signal)[0]

def valley_bounds(signal, peak_idx, minima_idx):
    left = minima_idx[minima_idx < peak_idx]
    right = minima_idx[minima_idx > peak_idx]
    if len(left) == 0 or len(right) == 0:
        return None, None
    return left[-1], right[0]

def strongest_extremum_near(signal, idx, window=25):
    lo = max(0, idx - window)
    hi = min(len(signal), idx + window)
    seg = signal[lo:hi]
    if len(seg) == 0: return idx
    i_max = lo + np.argmax(seg)
    i_min = lo + np.argmin(seg)
    return i_max

# ==================================================
# Custom Matplotlib Canvas
# ==================================================

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

# ==================================================
# Main Window
# ==================================================

class GCProcessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GC Data Processor with Calibration")
        self.resize(1400, 850)

        # State variables
        self.files = []              
        self.current_file_idx = -1   
        self.file_cache = {} 
        
        self.time = None
        self.signal = None
        self.peaks = []       
        self.bounds = []      
        self.peak_gases = []  
        
        self.selected_peak_idx = None 
        self.dragging = None 
        
        # Calibration State
        self.calib_df = pd.DataFrame()
        self.calib_filename = "None"

        self.init_ui()

    def init_ui(self):
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.splitter)

        # --- Left Panel ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_widget.setMinimumWidth(450)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        # 1. File Loader
        self.btn_load = QPushButton("📂 Load Files")
        self.btn_load.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6; color: white; padding: 10px; 
                font-weight: bold; border-radius: 5px; border: none;
            }
            QPushButton:hover { background-color: #2563EB; }
        """)
        self.btn_load.clicked.connect(self.load_files)
        left_layout.addWidget(self.btn_load)

        left_layout.addWidget(QLabel("File List:"))
        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self.on_file_select)
        left_layout.addWidget(self.file_list)

        # 2. Control Panel
        control_group = QFrame()
        control_group.setStyleSheet("QFrame { background-color: #2b2b2b; border-radius: 8px; } QLabel { color: #aaa; }")
        control_layout = QGridLayout(control_group)
        
        control_layout.addWidget(QLabel("EDIT TOOLS"), 0, 0, 1, 2)
        
        # Threshold
        lbl_sens = QLabel("THRESHOLD:")
        control_layout.addWidget(lbl_sens, 1, 0)
        self.spin_prominence = QDoubleSpinBox()
        self.spin_prominence.setRange(0.01, 1000.0)
        self.spin_prominence.setValue(1.0)
        self.spin_prominence.editingFinished.connect(self.recalc_peaks_from_spinbox)
        control_layout.addWidget(self.spin_prominence, 1, 1)

        # Checkboxes
        self.chk_add_mode = QCheckBox("➕ Add Peak")
        self.chk_add_mode.setStyleSheet("color: white;")
        self.chk_add_mode.toggled.connect(self.toggle_add_mode)
        control_layout.addWidget(self.chk_add_mode, 2, 0)

        self.chk_del_mode = QCheckBox("❌ Del Peak")
        self.chk_del_mode.setStyleSheet("color: white;")
        self.chk_del_mode.toggled.connect(self.toggle_del_mode)
        control_layout.addWidget(self.chk_del_mode, 2, 1)

        # Reset
        self.btn_reset = QPushButton("↺ Reset / Auto-Detect")
        self.btn_reset.setStyleSheet("background-color: #3B82F6; color: white; border-radius: 4px; padding: 6px;")
        self.btn_reset.clicked.connect(self.reset_current_processing)
        control_layout.addWidget(self.btn_reset, 3, 0, 1, 2)
        
        # Calibration Section
        control_layout.addWidget(QLabel("CALIBRATION"), 4, 0, 1, 2)
        
        self.lbl_calib_status = QLabel("File: None")
        self.lbl_calib_status.setStyleSheet("color: #FF6B6B; font-size: 10px;") # Red initially
        control_layout.addWidget(self.lbl_calib_status, 5, 0, 1, 2)

        self.btn_load_calib = QPushButton("Load Calibration File")
        self.btn_load_calib.setStyleSheet("background-color: #555; color: white; border-radius: 4px; padding: 4px;")
        self.btn_load_calib.clicked.connect(self.load_calibration_file)
        control_layout.addWidget(self.btn_load_calib, 6, 0, 1, 2)

        left_layout.addWidget(control_group)

        # 3. Results Table
        left_layout.addWidget(QLabel("Current Peaks:"))
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels(["RT", "Gas", "Area", "Amount (pmol)", "Height"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        left_layout.addWidget(self.result_table)

        # 4. Export
        self.btn_export = QPushButton("💾 Export All Excel")
        self.btn_export.setStyleSheet("background-color: #10B981; color: white; padding: 10px; border-radius: 5px;")
        self.btn_export.clicked.connect(self.export_results)
        left_layout.addWidget(self.btn_export)

        left_layout.setStretch(2, 1) 

        # --- Right Panel ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)

        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setSizes([450, 900])

    # ==================================================
    # Logic
    # ==================================================

    def load_files(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, "Open Data Files", "", "Text Files (*.txt *.csv)")
        if not fnames: return
        self.files = [Path(f) for f in sorted(fnames)]
        self.file_cache = {} 
        self.file_list.clear()
        self.file_list.addItems([f.name for f in self.files])
        self.file_list.setCurrentRow(0)

    def on_file_select(self, idx):
        if idx < 0 or idx >= len(self.files): return
        if self.current_file_idx != -1 and self.current_file_idx != idx:
            self.save_state_to_cache(self.current_file_idx)
        self.current_file_idx = idx
        self.load_file_data(idx)

    def load_file_data(self, idx):
        fname = self.files[idx].name
        if fname in self.file_cache:
            data = self.file_cache[fname]
            self.time = data['time']
            self.signal = data['signal']
            self.peaks = data['peaks']
            self.bounds = data['bounds']
            self.peak_gases = data.get('peak_gases', ["Unknown"] * len(self.peaks))
            if 'threshold' in data:
                self.spin_prominence.blockSignals(True)
                self.spin_prominence.setValue(data['threshold'])
                self.spin_prominence.blockSignals(False)
        else:
            try:
                path = self.files[idx]
                df = pd.read_csv(path, sep="\t")
                if df.shape[1] < 2: df = pd.read_csv(path, sep=",")
                self.time = df.iloc[:, 0].to_numpy()
                self.signal = df.iloc[:, 1].to_numpy()
                
                sig_range = np.max(self.signal) - np.min(self.signal)
                default_prom = max(0.1, sig_range * 0.01)
                self.spin_prominence.blockSignals(True)
                self.spin_prominence.setValue(default_prom)
                self.spin_prominence.blockSignals(False)
                
                self.auto_process_signal(fname, prominence=default_prom)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load {fname}:\n{e}")
                return

        self.selected_peak_idx = None
        self.refresh_plot(reset_view=True) 
        self.update_table()

    def save_state_to_cache(self, idx):
        if idx < 0 or idx >= len(self.files): return
        fname = self.files[idx].name
        self.file_cache[fname] = {
            'time': self.time,
            'signal': self.signal,
            'peaks': list(self.peaks),   
            'bounds': list(self.bounds),
            'peak_gases': list(self.peak_gases),
            'threshold': self.spin_prominence.value()
        }

    # --- Calibration Logic ---

    def load_calibration_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Calibration File", "", "Excel Files (*.xlsx)")
        if not path: return

        try:
            df = pd.read_excel(path)
            
            # Validation
            required_cols = {'Gas', 'RT_min', 'Tolerance', 'Slope', 'Intercept'}
            if not required_cols.issubset(df.columns):
                msg = ("Invalid Calibration File.\n\n"
                       "The Excel file MUST contain these columns:\n"
                       "- Gas (e.g., H2)\n"
                       "- RT_min (e.g., 2.89)\n"
                       "- Tolerance (e.g., 0.5)\n"
                       "- Slope\n"
                       "- Intercept\n\n"
                       "NOTE: The calculation assumes y = mx + c\n"
                       "Where:\n"
                       " y (Y-Axis) = Peak Area\n"
                       " x (X-Axis) = Amount (pmol)")
                QMessageBox.warning(self, "Validation Error", msg)
                return

            self.calib_df = df
            self.calib_filename = os.path.basename(path)
            
            # Update UI
            self.lbl_calib_status.setText(f"File: {self.calib_filename}")
            self.lbl_calib_status.setStyleSheet("color: #4CAF50; font-size: 10px;") # Green

            # Recalculate current file if loaded
            if self.time is not None:
                self.peak_gases = [self.identify_gas(self.time[p]) for p in self.peaks]
                self.update_table()
                self.refresh_plot(reset_view=False)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read Excel file:\n{e}")

    def identify_gas(self, rt):
        if self.calib_df.empty: return "Unknown"
        best_gas = "Unknown"
        min_diff = float('inf')
        for _, row in self.calib_df.iterrows():
            ref_rt = row['RT_min']
            tol = row['Tolerance']
            diff = abs(rt - ref_rt)
            if diff < tol and diff < min_diff:
                min_diff = diff
                best_gas = row['Gas']
        return best_gas

    def calculate_pmol(self, gas_name, area):
        # Returns 0 if no calib loaded or gas unknown
        if self.calib_df.empty: return 0.0
        row = self.calib_df[self.calib_df['Gas'] == gas_name]
        if row.empty: return 0.0
        
        slope = row.iloc[0]['Slope']
        intercept = row.iloc[0]['Intercept']
        if slope == 0: return 0.0
        
        # pmol = (Area - Intercept) / Slope
        return (area - intercept) / slope

    # --- Type Detection ---

    def get_file_type(self, fname):
        return "TCD" in fname.upper()

    def auto_process_signal(self, fname, prominence=None):
        if prominence is None:
            prominence = self.spin_prominence.value()

        is_tcd = self.get_file_type(fname)
        
        window = 21 if is_tcd else 31
        try:
            smooth = savgol_filter(self.signal, window, 3)
        except:
            smooth = self.signal
        raw_peaks = find_peaks(smooth, prominence=prominence)[0]
        
        if is_tcd:
            valid_peaks = [p for p in raw_peaks if self.time[p] > 2.0]
            raw_peaks = valid_peaks[:1] if valid_peaks else []
        else:
            raw_peaks = [p for p in raw_peaks if self.time[p] > 5.0]

        minima = find_valleys(self.signal)
        self.peaks = []
        self.bounds = []
        self.peak_gases = []
        
        for p in raw_peaks:
            l, r = valley_bounds(self.signal, p, minima)
            if l is not None and r is not None:
                self.peaks.append(p)
                self.bounds.append((l, r))
                self.peak_gases.append(self.identify_gas(self.time[p]))

    def recalc_peaks_from_spinbox(self):
        if self.current_file_idx == -1: return
        fname = self.files[self.current_file_idx].name
        val = self.spin_prominence.value()
        self.auto_process_signal(fname, prominence=val)
        self.selected_peak_idx = None
        self.refresh_plot(reset_view=False) 
        self.update_table()

    def reset_current_processing(self):
        if self.current_file_idx == -1: return
        sig_range = np.max(self.signal) - np.min(self.signal)
        default_prom = max(0.1, sig_range * 0.01)
        self.spin_prominence.setValue(default_prom)
        fname = self.files[self.current_file_idx].name
        self.auto_process_signal(fname, prominence=default_prom)
        self.selected_peak_idx = None 
        self.refresh_plot(reset_view=True)
        self.update_table()

    # ==================================================
    # Plotting & UI
    # ==================================================

    def refresh_plot(self, reset_view=False):
        xlim, ylim = None, None
        if not reset_view and self.canvas.ax.has_data():
            xlim = self.canvas.ax.get_xlim()
            ylim = self.canvas.ax.get_ylim()

        self.canvas.ax.clear()
        if self.time is None: 
            self.canvas.draw()
            return

        self.canvas.ax.plot(self.time, self.signal, 'k-', alpha=0.6, linewidth=1)

        for i, p in enumerate(self.peaks):
            color = 'r' if i == self.selected_peak_idx else 'b'
            size = 8 if i == self.selected_peak_idx else 5
            
            self.canvas.ax.plot(self.time[p], self.signal[p], marker='o', color=color, markersize=size)
            
            # Label
            gas_name = self.peak_gases[i] if i < len(self.peak_gases) else "?"
            label = f"{i+1}: {gas_name}"
            self.canvas.ax.text(self.time[p], self.signal[p], label, fontsize=9, verticalalignment='bottom')

            l, r = self.bounds[i]
            if l is not None and r is not None:
                baseline = np.linspace(self.signal[l], self.signal[r], r - l + 1)
                self.canvas.ax.fill_between(self.time[l:r + 1], baseline, self.signal[l:r + 1], color=color, alpha=0.2)
                self.canvas.ax.plot(self.time[l:r + 1], baseline, '--', color=color, linewidth=0.8)

        if self.selected_peak_idx is not None and self.selected_peak_idx < len(self.bounds):
            l, r = self.bounds[self.selected_peak_idx]
            self.canvas.ax.axvline(self.time[l], color='green', linestyle='--')
            self.canvas.ax.axvline(self.time[r], color='orange', linestyle='--')

        self.canvas.ax.set_xlabel("Time (min)")
        self.canvas.ax.set_ylabel("Signal")

        if not reset_view and xlim is not None:
            self.canvas.ax.set_xlim(xlim)
            self.canvas.ax.set_ylim(ylim)
        self.canvas.draw()

    def update_table(self):
        self.result_table.blockSignals(True)
        self.result_table.setRowCount(len(self.peaks))
        
        known_gases = ["Unknown"]
        if not self.calib_df.empty:
            known_gases += self.calib_df['Gas'].tolist()
        
        for i, (p, (l, r)) in enumerate(zip(self.peaks, self.bounds)):
            baseline = np.linspace(self.signal[l], self.signal[r], r - l + 1)
            area = trapezoid(self.signal[l:r + 1] - baseline, self.time[l:r + 1])
            height = abs(self.signal[p] - baseline[p-l])
            current_gas = self.peak_gases[i]
            
            self.result_table.setItem(i, 0, QTableWidgetItem(f"{self.time[p]:.3f}"))
            
            combo = QComboBox()
            combo.addItems(known_gases)
            index = combo.findText(current_gas)
            if index >= 0: combo.setCurrentIndex(index)
            else: combo.setCurrentIndex(0)
            combo.currentIndexChanged.connect(lambda idx, row=i: self.on_gas_change(row, idx))
            self.result_table.setCellWidget(i, 1, combo)

            self.result_table.setItem(i, 2, QTableWidgetItem(f"{area:.2f}"))
            
            pmol = self.calculate_pmol(current_gas, area)
            self.result_table.setItem(i, 3, QTableWidgetItem(f"{pmol:.2f}"))
            
            self.result_table.setItem(i, 4, QTableWidgetItem(f"{height:.2f}"))

        self.result_table.blockSignals(False)

    def on_gas_change(self, row, combo_idx):
        combo = self.result_table.cellWidget(row, 1)
        new_gas = combo.currentText()
        self.peak_gases[row] = new_gas
        
        l, r = self.bounds[row]
        baseline = np.linspace(self.signal[l], self.signal[r], r - l + 1)
        area = trapezoid(self.signal[l:r + 1] - baseline, self.time[l:r + 1])
        
        pmol = self.calculate_pmol(new_gas, area)
        self.result_table.setItem(row, 3, QTableWidgetItem(f"{pmol:.2f}"))
        self.refresh_plot(reset_view=False)

    # --- Mouse Interaction ---
    def toggle_add_mode(self, checked):
        if checked: self.chk_del_mode.setChecked(False)
        self.selected_peak_idx = None
        self.refresh_plot(reset_view=False)

    def toggle_del_mode(self, checked):
        if checked: self.chk_add_mode.setChecked(False)
        self.selected_peak_idx = None
        self.refresh_plot(reset_view=False)

    def on_click(self, event):
        if event.inaxes != self.canvas.ax or event.button != 1: return
        
        if self.chk_add_mode.isChecked():
            idx = int(np.argmin(np.abs(self.time - event.xdata)))
            lo = max(0, idx - 20)
            hi = min(len(self.signal), idx + 20)
            if hi > lo: idx = lo + np.argmax(self.signal[lo:hi])
            
            minima = find_valleys(self.signal)
            l, r = valley_bounds(self.signal, idx, minima)
            if l is None: l = max(0, idx - 50)
            if r is None: r = min(len(self.signal)-1, idx + 50)
            
            self.peaks.append(idx)
            self.bounds.append((l, r))
            self.peak_gases.append(self.identify_gas(self.time[idx]))
            
            sorted_zip = sorted(zip(self.peaks, self.bounds, self.peak_gases))
            self.peaks, self.bounds, self.peak_gases = zip(*sorted_zip)
            self.peaks, self.bounds, self.peak_gases = list(self.peaks), list(self.bounds), list(self.peak_gases)
            
            self.refresh_plot(reset_view=False)
            self.update_table()
            return

        if self.chk_del_mode.isChecked():
            click_x = event.xdata
            distances = [abs(self.time[p] - click_x) for p in self.peaks]
            if not distances: return
            closest = np.argmin(distances)
            if distances[closest] < (self.time[-1] - self.time[0]) * 0.05:
                self.peaks.pop(closest)
                self.bounds.pop(closest)
                self.peak_gases.pop(closest)
                self.refresh_plot(reset_view=False)
                self.update_table()
            return

        if self.selected_peak_idx is not None:
            l, r = self.bounds[self.selected_peak_idx]
            view_width = self.canvas.ax.get_xlim()[1] - self.canvas.ax.get_xlim()[0]
            tol_x = view_width * 0.02
            if abs(event.xdata - self.time[l]) < tol_x:
                self.dragging = 'left'; return
            elif abs(event.xdata - self.time[r]) < tol_x:
                self.dragging = 'right'; return

        click_x = event.xdata
        distances = [abs(self.time[p] - click_x) for p in self.peaks]
        if distances:
            closest = np.argmin(distances)
            view_width = self.canvas.ax.get_xlim()[1] - self.canvas.ax.get_xlim()[0]
            if distances[closest] < view_width * 0.1:
                self.selected_peak_idx = closest
                self.refresh_plot(reset_view=False)

    def on_motion(self, event):
        if self.dragging is None or event.xdata is None or self.selected_peak_idx is None: return
        idx = int(np.argmin(np.abs(self.time - event.xdata)))
        idx = max(0, min(len(self.time)-1, idx)) 
        peak_idx = self.peaks[self.selected_peak_idx]
        current_l, current_r = self.bounds[self.selected_peak_idx]

        if self.dragging == 'left' and idx < peak_idx:
            self.bounds[self.selected_peak_idx] = (idx, current_r)
        elif self.dragging == 'right' and idx > peak_idx:
            self.bounds[self.selected_peak_idx] = (current_l, idx)

        self.refresh_plot(reset_view=False)
        self.update_table()

    def on_release(self, event):
        self.dragging = None

    def export_results(self):
        if self.current_file_idx != -1: self.save_state_to_cache(self.current_file_idx)
        if not self.file_cache: return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Excel", "GC_Results.xlsx", "Excel Files (*.xlsx)")
        if not out_path: return

        final_rows = []
        for fname, data in self.file_cache.items():
            time, signal = data['time'], data['signal']
            peak_gases = data.get('peak_gases', [])
            detector = "TCD" if self.get_file_type(fname) else "FID"
            
            for i, (p, (l, r)) in enumerate(zip(data['peaks'], data['bounds'])):
                baseline = np.linspace(signal[l], signal[r], r - l + 1)
                area = trapezoid(signal[l:r + 1] - baseline, time[l:r + 1])
                gas_name = peak_gases[i] if i < len(peak_gases) else "Unknown"
                pmol = self.calculate_pmol(gas_name, area)
                
                final_rows.append({
                    "Filename": fname,
                    "Detector": detector,
                    "Gas": gas_name,
                    "Retention Time (min)": time[p],
                    "Peak Area": area,
                    "Amount (pmol)": pmol
                })

        try:
            pd.DataFrame(final_rows).to_excel(out_path, index=False)
            QMessageBox.information(self, "Success", f"Saved to {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"{e}")

def main():
    app = QApplication(sys.argv)
    window = GCProcessorGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
