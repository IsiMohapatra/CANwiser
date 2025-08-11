import customtkinter as ctk


def init_format_of_labels_page(app):
    page = ctk.CTkFrame(app, fg_color="#E8DFCA")
    app.pages["format_of_labels"] = page

    # Page Heading
    heading = ctk.CTkLabel(
        page,
        text="Select the Labels Format",
        font=ctk.CTkFont("Arial", 40, "bold"),
        text_color="#3E2723"
    )
    heading.pack(pady=(50, 20), anchor="center")

    # Frame to hold all label types and dropdowns
    dropdown_frame = ctk.CTkFrame(page, fg_color="transparent")
    dropdown_frame.pack(pady=30, anchor="w", padx=100)

    app.label_types = ["onfail", "init_type", "invld"]

    app.known_templates = {
        "onfail": ["{asw_signal}_{frame_id}hRcf_C", "{asw_signal}Dfl_C", "{asw_signal}Def_C"],
        "init_type": ["{asw_signal}_IniTyp_C", "{asw_signal}Ini_Type"],
        "invld": ["{asw_signal}_Invld_C", "{asw_signal}Invalid"]
    }

    app.selected_templates_by_type = {}
    app.template_entries = {}

    for row_idx, label_type in enumerate(app.label_types):
        label_text = format_label_type_display(label_type)

        ctk.CTkLabel(dropdown_frame, text="").grid(row=row_idx, column=0, padx=20)

        # Label
        ctk.CTkLabel(
            dropdown_frame,
            text=label_text,
            font=ctk.CTkFont("Arial", 18, "bold"),
            text_color="#6D4C41"
        ).grid(row=row_idx, column=1, sticky="w", padx=300)

        # Dropdown (ComboBox)
        entry = ctk.CTkComboBox(
            dropdown_frame,
            values=app.known_templates.get(label_type, []),
            fg_color="#8D6E63",
            button_color="#D9B382",
            button_hover_color="#D9B382",
            text_color="#FFFFFF",
            width=350,
            height=40
        )
        entry.set("")
        entry.grid(row=row_idx, column=1, padx=600, pady=20)

        app.template_entries[label_type] = entry

    # Validate and store function defined inside
    def validate_and_store_all_templates():
        for label_type, entry in app.template_entries.items():
            selected_template = entry.get().strip()

            if not selected_template:
                print(f"No format entered for {label_type}")
                continue

            app.selected_templates_by_type[label_type] = selected_template

            # Optionally update known list
            if selected_template not in app.known_templates[label_type]:
                app.known_templates[label_type].append(selected_template)

        from my_tool_gui.pages.frame_selection import init_frame_selection_page
        init_frame_selection_page(app)
        app.show_page("frame_selection")
    
    # Navigation Buttons
    button_frame = ctk.CTkFrame(page, fg_color="transparent")
    button_frame.pack(pady=30)

    ctk.CTkButton(
        button_frame,
        text="Next",
        width=120,
        fg_color="#C9A175",
        hover_color="#D9B382",
        text_color="#4E342E",
        command=validate_and_store_all_templates
    ).grid(row=0, column=1, padx=20, pady=30)

    ctk.CTkButton(
        button_frame,
        text="Back",
        width=120,
        fg_color="#C9A175",
        hover_color="#D9B382",
        text_color="#4E342E",
        command=lambda: app.show_page("input")
    ).grid(row=0, column=0, padx=20, pady=30)

    app.add_footer(page)


def format_label_type_display(label_type):
    mapping = {
        "onfail": "OnFail/FailSafe Calibration",
        "init_type": "Init_Type Calibration",
        "invld": "Invalid Calibration"
    }
    return mapping.get(label_type, label_type.capitalize())
