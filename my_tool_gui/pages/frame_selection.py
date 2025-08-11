import customtkinter as ctk
import tkinter as tk
import os
from tkinter import ttk
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from my_tool.data_loading import load_signals_from_excel



def init_frame_selection_page(app):
    page = ctk.CTkFrame(app, fg_color="#E8DFCA")
    app.pages["frame_selection"] = page

    # Initialize dynamic widgets list
    app.dynamic_widgets = []

    box = ctk.CTkFrame(page, fg_color="#8D6E63", corner_radius=9, width=1500)
    box.pack(pady=80, padx=20, anchor="center")

    # Question Label
    question_label = ctk.CTkLabel(
        box,
        text="Analysis required for all the Signals present in the\nmeasurement/CAN Matrix?",
        font=ctk.CTkFont("Arial", 24, "bold"),
        text_color="#FFFFFF",
        justify="center"
    )
    question_label.pack(pady=(20, 15), padx=20)

    # Button frame
    btn_frame = ctk.CTkFrame(box, fg_color="transparent")
    btn_frame.pack(pady=10)

    app.analysis_status = "Yes"

    yes_btn = ctk.CTkButton(
        btn_frame, text="Yes", width=100, fg_color="#fff", text_color="#000",
        hover_color="#ccc", command=lambda: full_signal_analysis(app,page)
    )
    yes_btn.pack(side="left", padx=20)

    no_btn = ctk.CTkButton(
        btn_frame, text="No", width=100, fg_color="#fff", text_color="#000",
        hover_color="#ccc", command=lambda: reveal_frame_input(app,page)
    )
    no_btn.pack(side="right", padx=20)

    back_btn = ctk.CTkButton(page, text="Back", width=120, fg_color="#C9A175", hover_color="#D9B382", text_color="#4E342E", 
                             command=lambda: app.show_page("format_of_labels"))
    back_btn.place(x=40, rely=1.0, y=-50, anchor="sw")

    next_btn = ctk.CTkButton(page, text="Next", width=120, fg_color="#C9A175", hover_color="#D9B382", text_color="#4E342E", 
                             command=handle_next_button(app))
    next_btn.place(relx=1.0, rely=1.0, x=-40, y=-50, anchor="se")

    app.add_footer(page)


def handle_next_button(app):
    recording_type = app.selected_type_of_recording

    if recording_type == "CANalyser Plot/Recording":
        app.show_page("sync")

    elif recording_type == "CAN monitoring/recording":
        folder_path = app.folder_entry.get().strip()

        if not folder_path.lower().endswith(".mf4") or not Path(folder_path).is_file():
            tk.messagebox.showwarning("Invalid File", "Please select a valid .mf4 file before proceeding.")
            return

        print("Performing preprocessing for CAN monitoring...")
        app.show_page("processing")

    else:
        tk.messagebox.showwarning("Recording Type Not Selected", "Please select a type of recording before continuing.")


def full_signal_analysis(app, page):
    app.analysis_status = "Yes"
    for widget in getattr(app, "dynamic_widgets", []):
        widget.destroy()
    app.dynamic_widgets.clear()

    can_matrix_path = app.saved_files.get("can_matrix", "")
    if os.path.exists(can_matrix_path):
        app.tx_df, app.rx_df = load_signals_from_excel(can_matrix_path)
        if app.tx_df is not None and app.rx_df is not None:
            app.frame_names = extract_frame_names(app)

    app.selected_frames = app.frame_names.copy()

    output_section = ctk.CTkFrame(page, fg_color="#E8DFCA", corner_radius=10)
    output_section.pack(pady=10, padx=20, fill="x")
    app.dynamic_widgets.append(output_section)

    ctk.CTkLabel(output_section, text="CAN Frame Overview",
                 font=ctk.CTkFont("Arial", 15, weight="bold"), text_color="#3E3E3E").pack(fill="x", pady=(10, 10))

    container = ctk.CTkFrame(output_section, fg_color="#F5F1E6", corner_radius=12)
    container.pack(padx=15, pady=(0, 15))
    app.dynamic_widgets.append(container)

    canvas = tk.Canvas(container, width=1500, height=400, highlightthickness=0)
    canvas.pack(side="left")
    vscrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    vscrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=vscrollbar.set)

    frame_inside_canvas = ctk.CTkFrame(canvas, fg_color="#F5F1E6")
    canvas.create_window((0, 0), window=frame_inside_canvas, anchor="nw")

    for i, frame in enumerate(app.frame_names):
        row = i // 6
        col = i % 6
        label = ctk.CTkLabel(frame_inside_canvas, text=frame, font=ctk.CTkFont("Arial", 11),
                             text_color="#4B4B4B", fg_color="#DACBBE", corner_radius=14, padx=8, pady=3)
        label.grid(row=row, column=col, padx=4, pady=4, sticky="w")
        app.dynamic_widgets.append(label)

    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    frame_inside_canvas.bind("<Configure>", on_frame_configure)

    if hasattr(app, "suggestion_frame"):
        app.suggestion_frame.pack_forget()


def reveal_frame_input(self, page):
    for widget in getattr(self, "dynamic_widgets", []):
        widget.destroy()
    self.dynamic_widgets.clear()

    self.analysis_status = "No"
    self.frame_input_section = ctk.CTkFrame(page, fg_color="transparent")
    self.frame_input_section.pack(pady=20)
    self.dynamic_widgets.append(self.frame_input_section)

    label_title = ctk.CTkLabel(
        self.frame_input_section, text="🔍 Input Frame(s) for Analysis.",
        font=ctk.CTkFont("Arial", 24, "bold"), text_color="#3E2723"
    )
    label_title.pack(pady=(10, 10))
    self.dynamic_widgets.append(label_title)

    content_frame = ctk.CTkFrame(self.frame_input_section, fg_color="transparent")
    content_frame.pack(pady=10)
    self.dynamic_widgets.append(content_frame)

    can_matrix_path = self.saved_files.get("can_matrix", "")
    if os.path.exists(can_matrix_path):
        try:
            self.tx_df, self.rx_df = load_signals_from_excel(can_matrix_path)
            if self.tx_df is not None and self.rx_df is not None:
                from my_tool_gui.widgets.suggestion import Trie
                self.trie = Trie()
                self.frame_names = extract_frame_names(self)
                for frame_name in self.frame_names:
                    self.trie.insert(frame_name)

                label_input = ctk.CTkLabel(content_frame, text="Enter the frame for Analysis:\nPlease Press Enter to see the selected frames.",
                                           font=ctk.CTkFont("Arial", 18), text_color="#6D4C41", anchor="e", width=220)
                label_input.grid(row=0, column=0, padx=(20, 10), pady=10, sticky="e")
                self.dynamic_widgets.append(label_input)

                self.frame_input = ctk.CTkEntry(content_frame, width=500, height=60, font=ctk.CTkFont("Arial", 14), justify="left")
                self.frame_input.grid(row=0, column=1, padx=(0, 20), pady=10)
                self.dynamic_widgets.append(self.frame_input)

                self.frame_input.bind("<KeyRelease>", lambda event: search_and_suggest_frames(self, event))
                self.frame_input.bind("<Return>", lambda event: display_selected_frames(self, event))
                self.frame_input.bind("<FocusIn>", lambda event: search_and_suggest_frames(self, event))

                self.suggestion_frame = ctk.CTkFrame(page, width=500, height=200, fg_color="transparent")
                self.suggestion_frame.pack(pady=(5, 10), padx=10, anchor="center")
                self.suggestion_frame.pack_forget()
                self.dynamic_widgets.append(self.suggestion_frame)

                label_output = ctk.CTkLabel(content_frame, text="Your selected Frame(s) are:",
                                            font=ctk.CTkFont("Arial", 18), text_color="#6D4C41",
                                            anchor="e", width=220)
                label_output.grid(row=1, column=0, padx=(10, 10), pady=15, sticky="e")
                self.dynamic_widgets.append(label_output)

                self.selected_frames_display = ctk.CTkLabel(content_frame, text="", font=ctk.CTkFont("Arial", 18), text_color="#6D4C41")
                self.selected_frames_display.grid(row=1, column=1, padx=(0, 20), pady=10, sticky="w")
                self.dynamic_widgets.append(self.selected_frames_display)

        except Exception as e:
            self.sync_error_label = ctk.CTkLabel(page, text=f"❌ Error loading CAN Matrix: {str(e)}",
                                                 font=("Arial", 14, "bold"), fg_color="#FF6B6B",
                                                 text_color="black", width=600)
            self.sync_error_label.pack(pady=(10, 10))
            self.dynamic_widgets.append(self.sync_error_label)


def extract_frame_names(app):
    tx_signals_frame = app.tx_df['Frame Name'].dropna().unique().tolist() if 'Frame Name' in app.tx_df.columns else []
    rx_signals_frame = app.rx_df['Frame Name'].dropna().unique().tolist() if 'Frame Name' in app.rx_df.columns else []
    return list(set(tx_signals_frame + rx_signals_frame))


def display_selected_frames(self, event=None):
    if self.suggestion_frame:
        self.suggestion_frame.pack_forget()
        destroy_suggestions(self)

    user_input = self.frame_input.get()
    if user_input:
        new_frames = [frame.strip() for frame in user_input.split(",") if frame.strip()]
        self.selected_frames = new_frames
        self.selected_frames_display.configure(text=", ".join(self.selected_frames))

    self.frame_input.master.focus()


def search_and_suggest_frames(self, event=None):
    typed_text = self.frame_input.get().strip()

    if not typed_text:
        if self.suggestion_frame:
            self.suggestion_frame.pack_forget()
        return

    frames = [frame.strip() for frame in typed_text.split(",")]
    current_frame = frames[-1] if frames else ""

    if not current_frame:
        if self.suggestion_frame:
            self.suggestion_frame.pack_forget()
        return

    matching_frames = [frame for frame in self.trie.get_all_frames()
                       if current_frame.lower() in frame.lower()]

    destroy_suggestions(self)

    if matching_frames:
        for frame in matching_frames:
            frame_option = ctk.CTkButton(
                self.suggestion_frame,
                text=frame,
                fg_color="#E0E0E0",
                hover_color="#D0D0D0",
                text_color="#000000",
                font=("Arial", 12),
                command=lambda s=frame: add_frame_to_input(self,s)
            )
            frame_option.pack(padx=10, anchor="center")
    else:
        no_match_frame = ctk.CTkLabel(
            self.suggestion_frame,
            text="No matches found",
            font=("Arial", 12),
            text_color="gray"
        )
        no_match_frame.pack(padx=10, anchor="center")

    if self.suggestion_frame and not self.suggestion_frame.winfo_ismapped():
        self.suggestion_frame.pack(pady=(5, 10), padx=10, anchor="center")


def add_frame_to_input(self, frame_name):
    typed_text = self.frame_input.get().strip()
    frames = [frame.strip() for frame in typed_text.split(",")]
    frames[-1] = frame_name
    self.frame_input.delete(0, "end")
    self.frame_input.insert(0, ", ".join(frames))
    self.suggestion_frame.pack_forget()


def destroy_suggestions(self):
    if self.suggestion_frame is not None:
        for widget in self.suggestion_frame.winfo_children():
            widget.destroy()
