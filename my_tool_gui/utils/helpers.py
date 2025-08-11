import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import os, shutil, tempfile
import sys
import customtkinter as ctk
from tkinter import filedialog, messagebox


def download_template(template_type):
    templates = {
        "can_matrix": "can_matrix_template.xlsx",
        "dtc_matrix": "dtc_matrix_template.xlsx",
        "gateway": "gateway_template.xlsx",
        "fid": "fid_mapping_template.xlsx",
        "report": "report_template.xlsx"
    }

    template_name = templates.get(template_type)
    if not template_name:
        messagebox.showerror("Error", "Template type not recognized.")
        return

    try:
        # Path to template inside bundled app or dev mode
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        template_path = os.path.join(base_path, "templates", template_name)

        # Define auto-download location: user's Downloads folder
        downloads_folder = str(Path.home() / "Downloads")
        os.makedirs(downloads_folder, exist_ok=True)
        save_path = os.path.join(downloads_folder, template_name)
        shutil.copy(template_path, save_path)
        messagebox.showinfo("Success", f"Template downloaded to:\n{save_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save template:\n{e}")

def select_can(app):
    file = filedialog.askopenfilename()
    if file:
        app.can_entry.delete(0, "end")
        app.can_entry.insert(0, file)
        print(f"Selected CAN Matrix Path: {file}")

def select_dtc(app):
    file = filedialog.askopenfilename()
    if file:
        app.dtc_entry.delete(0, "end")
        app.dtc_entry.insert(0, file)
        print(f"Selected DTC Matrix Path: {file}")

def select_gateway(app):
    file = filedialog.askopenfilename()
    if file:
        app.gateway_entry.delete(0, "end")
        app.gateway_entry.insert(0, file)
        print(f"Selected Gateway Sheet Path: {file}")

def select_fid(app):
    file = filedialog.askopenfilename()
    if file:
        app.fid_entry.delete(0, "end")
        app.fid_entry.insert(0, file)
        print(f"Selected FID Mapping Path: {file}")

def select_report_template(app):
    file = filedialog.askopenfilename()
    if file:
        app.report_entry.delete(0, "end")
        app.report_entry.insert(0, file)
        app.report_path_original = file
        print(f"Selected Report Tempelate Path: {file}")

def select_folder(app):
    dialog = tk.Toplevel(app)
    dialog.title("Select Type")
    dialog.geometry("300x120")
    dialog.transient(app)
    dialog.grab_set()

    label = tk.Label(dialog, text="Choose what to select:", font=("Arial", 12))
    label.pack(pady=10)

    def select_file():
        path = filedialog.askopenfilename(title="Select File")
        if path:
            app.folder_entry.delete(0, "end")
            app.folder_entry.insert(0, path)
            app.folder_valid = True
            if app.folder_error_label:
                app.folder_error_label.destroy()
            print(f"Selected File Path: {path}")
        dialog.destroy()

    def select_folder():
        path = filedialog.askdirectory(title="Select Folder")
        if path:
            app.folder_entry.delete(0, "end")
            app.folder_entry.insert(0, path)
            app.folder_valid = True
            if app.folder_error_label:
                app.folder_error_label.destroy()
            print(f"Selected Folder Path: {path}")
        dialog.destroy()

    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=10)

    btn_file = tk.Button(btn_frame, text="Select File", width=12, command=select_file)
    btn_file.pack(side="left", padx=10)

    btn_folder = tk.Button(btn_frame, text="Select Folder", width=12, command=select_folder)
    btn_folder.pack(side="right", padx=10)

    # Center the dialog properly **after it's fully rendered**
    dialog.update_idletasks()
    screen_width = dialog.winfo_screenwidth()
    screen_height = dialog.winfo_screenheight()
    window_width = dialog.winfo_width()
    window_height = dialog.winfo_height()
    x = (screen_width // 2) - (window_width // 2) + 150 
    y = (screen_height // 2) - (window_height // 2) + 100
    dialog.geometry(f"{window_width}x{window_height}+{x}+{y}")

def validate_and_go_to_sync(app):
    can_path = app.can_entry.get().strip()
    dtc_path = app.dtc_entry.get().strip()
    gateway_path = app.gateway_entry.get().strip()
    fid_path = app.fid_entry.get().strip()
    folder_path = app.folder_entry.get().strip()

    if not can_path:
        messagebox.showwarning("Missing Input", "Please load the CAN Matrix file.")
        return

    if not folder_path:
        messagebox.showwarning("Missing Input", "Please provide a measurement path.")
        return

    if app.selected_type_of_recording == "CANalyser Plot/Recording":
        folder = Path(folder_path)
        if not folder.is_dir() or not any(f.is_dir() for f in folder.iterdir()):
            messagebox.showwarning("Invalid Folder", "Selected folder must contain subfolders.")
            return
        app.folder_valid = True

    elif app.selected_type_of_recording == "CAN monitoring/recording":
        if not folder_path.lower().endswith(".mf4") or not Path(folder_path).is_file():
            messagebox.showwarning("Invalid File", "Please select a valid .mf4 measurement file.")
            return
        app.folder_valid = True

    try:
        os.makedirs(app.temp_dir, exist_ok=True)

        can_dest = shutil.copy(can_path, os.path.join(app.temp_dir, os.path.basename(can_path)))
        dtc_dest = shutil.copy(dtc_path, os.path.join(app.temp_dir, os.path.basename(dtc_path))) if dtc_path else None
        gateway_dest = shutil.copy(gateway_path, os.path.join(app.temp_dir, os.path.basename(gateway_path))) if gateway_path else None
        fid_dest = shutil.copy(fid_path, os.path.join(app.temp_dir, os.path.basename(fid_path))) if fid_path else None

        app.saved_files = {
            "folder_path": folder_path,
            "can_matrix": can_dest,
            "dtc_matrix": dtc_dest,
            "gateway_sheet": gateway_dest,
            "fid_sheet": fid_dest
        }

        if app.function_status.lower() == "report generation":
            report_path = app.report_entry.get().strip()
            if report_path:
                app.saved_files["report_path"] = report_path

        app.sync_status = "success"
        from my_tool_gui.pages.format_of_labels import init_format_of_labels_page
        init_format_of_labels_page(app)
        app.show_page("format_of_labels")

    except Exception as e:
        print("Error saving files:", e)
        app.sync_status = "error"
