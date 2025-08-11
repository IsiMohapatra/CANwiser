import customtkinter as ctk
from my_tool_gui.widgets.carousel import setup_carousel

def init_main_page(app):
    page = ctk.CTkFrame(app, fg_color="#E8DFCA")
    app.pages["main"] = page
    page.pack(fill="both", expand=True)

    page.columnconfigure(0, weight=1)
    page.columnconfigure(1, weight=1)
    page.rowconfigure(0, weight=0)
    page.rowconfigure(1, weight=1)

    title_label = ctk.CTkLabel(
        page, text="WELCOME TO CANWiser",
        font=("Montserrat Black", 60, "bold"),
        text_color="#3E2723"
    )
    title_label.place(relx=0.5, y=100, anchor="n")

    subtitle_label = ctk.CTkLabel(
        page, text="Your intelligent companion for CAN signal analysis & reporting.",
        font=("Lato Italic", 20, "italic"),
        text_color="#6D4C41"
    )
    subtitle_label.grid(row=0, column=0, columnspan=2, pady=(180, 0), sticky="n")

    left_frame = ctk.CTkFrame(page, fg_color="transparent")
    left_frame.grid(row=1, column=0, sticky="nsew")
    left_frame.grid_rowconfigure(0, weight=1)
    left_frame.grid_rowconfigure(2, weight=1)
    left_frame.grid_columnconfigure(0, weight=1)

    btn_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
    btn_frame.grid(row=0, column=0, pady=(50, 0))

    analysis_button = ctk.CTkButton(
        btn_frame, text="🚦   Measurement Analysis",
        font=("Inter", 24, "bold"), fg_color="#6D4C41", hover_color="#7a9669",
        text_color="#FFFFFF", corner_radius=12, height=80, width=500,
        command=lambda: set_function_status_and_redirect(app, "Measurement Analysis")
    )
    analysis_button.pack(pady=(10, 10))

    report_button = ctk.CTkButton(
        btn_frame, text="📄  Generate Report",
        font=("Inter", 24, "bold"), fg_color="#6D4C41", hover_color="#7a9669",
        text_color="#FFFFFF", corner_radius=12, height=80, width=500,
        command=lambda: set_function_status_and_redirect(app, "Report Generation")
    )
    report_button.pack(pady=(25, 10))

    right_frame = ctk.CTkFrame(page, fg_color="transparent")
    right_frame.grid(row=1, column=1, sticky="nsew")

    # Carousel display
    setup_carousel(app, right_frame)


def set_function_status_and_redirect(app, status):
    app.function_status = status
    print("Function Status Set:", app.function_status)

    # Ensure the input page is reset if user goes back
    app.pages.pop("input", None)

    # Dynamically re-import and init the input page
    from my_tool_gui.pages.input_page import init_input_page
    init_input_page(app)
    app.show_page("input")  