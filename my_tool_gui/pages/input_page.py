import customtkinter as ctk
from my_tool_gui.utils.helpers import (
    select_can, select_dtc, select_gateway, select_fid, select_folder,select_report_template,
    download_template, validate_and_go_to_sync)

def init_input_page(app):
    page = ctk.CTkFrame(app, fg_color="#E8DFCA")
    app.pages["input"] = page

    # Title based on function status
    title = "Measurement Analysis" if app.function_status == "Measurement Analysis" else "Report Generation"
    ctk.CTkLabel(page, text=title, font=("Arial", 40, "bold"), text_color="#3E2723").pack(pady=(50, 20), anchor="center")

    ctk.CTkLabel(page, text="Select the Input Sheets", font=("Arial", 20), text_color="#6D4C41").pack(pady=(10, 20), anchor="center")

    frame = ctk.CTkFrame(page, fg_color="transparent")
    frame.pack(pady=5, anchor="w", padx=30)

    # CAN MATRIX
    app.can_entry = ctk.CTkEntry(frame, width=500, height=40, placeholder_text="Select CAN Matrix")
    app.can_entry.grid(row=0, column=1, padx=10, pady=15)
    ctk.CTkButton(frame, text="📄", width=40, fg_color="#D7CCC8", hover_color="#BCAAA4", text_color="#4E342E", command=lambda: select_can(app)).grid(row=0, column=2, padx=5)
    ctk.CTkLabel(frame, text="CAN MATRIX", font=("Arial", 18, "bold"), text_color="#6D4C41").grid(row=0, column=0, sticky="w", padx=(200, 5))
    ctk.CTkButton(frame, text="📥 Download CAN Matrix Template", width=40, fg_color="#8D6E63", hover_color="#795548", text_color="#FFFFFF",
                  command=lambda: download_template(app, "can_matrix")).grid(row=0, column=3, padx=10)

    # DTC MATRIX
    app.dtc_entry = ctk.CTkEntry(frame, width=500, height=40, placeholder_text="Select DTC Matrix")
    app.dtc_entry.grid(row=1, column=1, padx=50, pady=5)
    ctk.CTkButton(frame, text="📄", width=40, fg_color="#D7CCC8", hover_color="#BCAAA4", text_color="#4E342E", command=lambda: select_dtc(app)).grid(row=1, column=2, padx=5)
    ctk.CTkLabel(frame, text="DTC MATRIX", font=("Arial", 18, "bold"), text_color="#6D4C41").grid(row=1, column=0, sticky="w", padx=(200, 5))
    ctk.CTkButton(frame, text="📥 Download DTC Matrix Template", width=40, fg_color="#8D6E63", hover_color="#795548", text_color="#FFFFFF",
                  command=lambda: download_template(app, "dtc_matrix")).grid(row=1, column=3, padx=10)

    # GATEWAY
    app.gateway_entry = ctk.CTkEntry(frame, width=500, height=40, placeholder_text="Select Gateway Sheet")
    app.gateway_entry.grid(row=2, column=1, padx=50, pady=15)
    ctk.CTkButton(frame, text="📄", width=40, fg_color="#D7CCC8", hover_color="#BCAAA4", text_color="#4E342E", command=lambda: select_gateway(app)).grid(row=2, column=2, padx=5)
    ctk.CTkLabel(frame, text="GATEWAY SHEET", font=("Arial", 18, "bold"), text_color="#6D4C41").grid(row=2, column=0, sticky="w", padx=(200, 5))
    ctk.CTkButton(frame, text="📥 Download Gateway Sheet Template", width=40, fg_color="#8D6E63", hover_color="#795548", text_color="#FFFFFF",
                  command=lambda: download_template(app, "gateway")).grid(row=2, column=3, padx=10)

    # FID MAPPING
    app.fid_entry = ctk.CTkEntry(frame, width=500, height=40, placeholder_text="Select FID Mapping")
    app.fid_entry.grid(row=3, column=1, padx=50, pady=15)
    ctk.CTkButton(frame, text="📄", width=40, fg_color="#D7CCC8", hover_color="#BCAAA4", text_color="#4E342E", command=lambda: select_fid(app)).grid(row=3, column=2, padx=5)
    ctk.CTkLabel(frame, text="FID MAPPING", font=("Arial", 18, "bold"), text_color="#6D4C41").grid(row=3, column=0, sticky="w", padx=(200, 5))
    ctk.CTkButton(frame, text="📥 Download Fid Mapping Template", width=40, fg_color="#8D6E63", hover_color="#795548", text_color="#FFFFFF",
                  command=lambda: download_template(app, "fid")).grid(row=3, column=3, padx=10)

    # MEASUREMENTS
    app.folder_entry = ctk.CTkEntry(frame, width=500, height=40, placeholder_text="Select Measurement Folder")
    app.folder_entry.grid(row=4, column=1, padx=50, pady=15)
    ctk.CTkButton(frame, text="📁", width=40, fg_color="#D7CCC8", hover_color="#BCAAA4", text_color="#4E342E", command=lambda: select_folder(app)).grid(row=4, column=2, padx=5)
    ctk.CTkLabel(frame, text="MEASUREMENTS FOLDER", font=("Arial", 18, "bold"), text_color="#6D4C41").grid(row=4, column=0, sticky="w", padx=(200, 5))

    # REPORT TEMPLATE (optional)
    if app.function_status.lower() == "report generation":
        app.report_entry = ctk.CTkEntry(frame, width=500, height=40, placeholder_text="Select Report Template")
        app.report_entry.grid(row=5, column=1, padx=50, pady=15)
        ctk.CTkButton(frame, text="📄", width=40, fg_color="#D7CCC8", hover_color="#BCAAA4", text_color="#4E342E", command=lambda: select_report_template(app)).grid(row=5, column=2, padx=5)
        ctk.CTkLabel(frame, text="REPORT TEMPLATE", font=("Arial", 18, "bold"), text_color="#6D4C41").grid(row=5, column=0, sticky="w", padx=(200, 5))
        ctk.CTkButton(frame, text="📥 Download Report Template", width=40, fg_color="#8D6E63", hover_color="#795548", text_color="#FFFFFF",
                      command=lambda: download_template(app, "report")).grid(row=5, column=3, padx=10)

    # Dropdown
    app.drop_menu = ctk.CTkOptionMenu(page, values=["CANalyser Plot/Recording", "HIL recording", "CAN monitoring/recording"],
                                      fg_color="#8D6E63", button_color="#D9B382", button_hover_color="#D9B382", text_color="#FFFFFF")
    app.drop_menu.set("Select Type of Recording")
    app.drop_menu.pack(pady=(30, 10))

    # Navigation
    ctk.CTkButton(page, text="Next", width=120, fg_color="#C9A175", hover_color="#D9B382", text_color="#4E342E",
                  command=lambda: check_selection_and_proceed(app)).pack(pady=10)

    ctk.CTkButton(page, text="Back", width=120, fg_color="#C9A175", hover_color="#D9B382", text_color="#4E342E",
                  command=lambda: app.show_page("main")).pack(pady=(10, 30))

    app.add_footer(page)

    
def check_selection_and_proceed(app):
    selected = app.drop_menu.get()
    if selected == "CANalyser Plot/Recording":
        app.selected_type_of_recording = selected
        validate_and_go_to_sync(app)
        print(app.selected_type_of_recording)
    if selected == "CAN monitoring/recording":
        app.selected_type_of_recording = selected
        validate_and_go_to_sync(app)
        print(app.selected_type_of_recording)