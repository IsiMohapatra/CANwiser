import customtkinter as ctk
import tkinter as tk
import tempfile

# Page imports
from my_tool_gui.pages.intial_page import init_main_page
from my_tool_gui.pages.input_page import init_input_page
from my_tool_gui.pages.format_of_labels import init_format_of_labels_page
from my_tool_gui.pages.frame_selection import init_frame_selection_page
# from my_tool_gui.pages.sync_page import init_sync_page
# from my_tool_gui.pages.processing_page import init_processing_page

# UI Theme Setup
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class CANwiserApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CANwiser")
        self.geometry("900x650")
        self.configure(fg_color="#E8DFCA")

        # App state
        self.pages = {}
        self.function_status = "Idle"
        self.saved_files = {}
        self.folder_valid = False
        self.folder_error_label = None 
        self.temp_dir = tempfile.mkdtemp()
        self.dynamic_widgets = []

        # Initialize pages
        init_main_page(self)
        init_input_page(self)
        init_format_of_labels_page(self)
        init_frame_selection_page(self)
        # init_sync_page(self)
        # init_processing_page(self)

        self.show_page("main")  # Show the initial page
        self.after(100, lambda: self.state("zoomed"))  # Start maximized

    def show_page(self, name):
        if name not in self.pages:
            print(f"[ERROR] Page '{name}' not found in self.pages")
            return

        current_page = None
        loading_label = None

        for page in self.pages.values():
            if page.winfo_ismapped():
                current_page = page
                break

        if current_page:
            button_widgets = [
                widget for widget in current_page.winfo_children()
                if isinstance(widget, ctk.CTkButton)
            ]
            next_button = next(
                (btn for btn in button_widgets if btn.cget("text") == "Next"), None
            )

            if next_button:
                loading_label = tk.Label(
                    current_page,
                    text="Loading.....",
                    font=("Arial", 14),
                    fg="black",
                    bg=current_page["bg"]
                )
                loading_label.place(
                    x=next_button.winfo_x(),
                    y=next_button.winfo_y() + next_button.winfo_height() + 5
                )
                self.update_idletasks()

        # Recreate dynamic pages if needed
        if name == "input":
            init_input_page(self)
        elif name == "format_of_labels":
            init_format_of_labels_page(self)
        elif name == "frame_selection":
            init_frame_selection_page(self)
        elif name == "sync":
            self.init_sync_page(self)
        elif name == "processing":
            self.init_processing_page(self)
        

        # Hide all pages
        for page in self.pages.values():
            page.pack_forget()

        # Show the requested page
        if loading_label:
            loading_label.destroy()
        self.pages[name].pack(fill="both", expand=True)

    def add_footer(self, page):
        """Optional footer for all pages."""
        footer = ctk.CTkLabel(
            page,
            text="CANwiser © Bosch",
            text_color="#9E9E9E",
            font=("Arial", 10),
            anchor="center"
        )
        footer.pack(side="bottom", pady=5)

    
