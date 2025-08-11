from PIL import ImageTk, ImageEnhance, ImageOps, ImageDraw
import tkinter as tk
from PIL import Image


# -----------------------------------------------------------------------------------------
# ---------------  CAROUSEL SETUP FOR THE 1ST PAGE ----------------------------------------
# -----------------------------------------------------------------------------------------


def setup_carousel(app, right_frame):
    canvas_width, canvas_height = 860, 620
    app.canvas = tk.Canvas(
        right_frame, width=canvas_width, height=canvas_height,
        bg="#E8DFCA", bd=0, highlightthickness=0, relief='ridge'
    )
    app.canvas.pack()

    app.card_bg_img = draw_rounded_card(canvas_width, canvas_height)
    app.canvas.create_image(0, 0, anchor="nw", image=app.card_bg_img)

    image_files = ["frame1.png", "frame2.png", "frame3.png"]
    app.images_color = [Image.open(f).convert("RGBA") for f in image_files]
    app.images_color = [ImageOps.expand(img, border=10, fill="#BDBDBD") for img in app.images_color]
    app.images_center = [img.resize((600, 400), Image.Resampling.LANCZOS) for img in app.images_color]

    app.images_side = []
    for img in app.images_color:
        gray = ImageOps.grayscale(img.resize((500, 300), Image.Resampling.LANCZOS)).convert("RGBA")
        faded = ImageEnhance.Brightness(gray).enhance(0.3)
        app.images_side.append(faded)

    app.current_index = 0
    app.image_refs = {}

    dot_frame = tk.Frame(right_frame, bg="#E8DFCA")
    dot_frame.place(x=400, y=450)
    app.dots = []
    for i in range(len(app.images_color)):
        dot = tk.Label(dot_frame, text="●", fg="#BDBDBD", font=("Arial", 16), bg="#E8DFCA")
        dot.pack(side="left", padx=4)
        app.dots.append(dot)
    app.dots[0].configure(fg="#212121")

    draw_carousel(app)
    app.after(3000, lambda: animate_transition(app))

def draw_rounded_card(width, height, radius=30, color="#E8DFCA"):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([(0, 0), (width, height)], radius=radius, fill=color)
    return ImageTk.PhotoImage(img)

def draw_carousel(app):
    app.canvas.delete("carousel")
    idx = app.current_index
    total = len(app.images_color)
    left = (idx - 1) % total
    center = idx
    right = (idx + 1) % total

    img_left = ImageTk.PhotoImage(app.images_side[left])
    app.canvas.create_image(50, 100, anchor="nw", image=img_left, tags="carousel")

    img_right = ImageTk.PhotoImage(app.images_side[right])
    app.canvas.create_image(400, 100, anchor="nw", image=img_right, tags="carousel")

    app.canvas.create_rectangle(140, 50, 760, 470, fill="#D6CEC2", outline="", tags="carousel")

    img_center = ImageTk.PhotoImage(app.images_center[center])
    app.canvas.create_image(150, 60, anchor="nw", image=img_center, tags="carousel")

    app.image_refs = {"left": img_left, "center": img_center, "right": img_right}
    update_dots(app)

def animate_transition(app):
    steps = 15
    delay = 20

    idx = app.current_index
    total = len(app.images_color)
    left = (idx - 1) % total
    center = idx
    right = (idx + 1) % total
    next_right = (idx + 2) % total

    center_img = app.images_center[center]
    right_img = app.images_side[right]
    next_img = app.images_side[next_right]

    for step in range(steps + 1):
        app.canvas.delete("carousel")
        t = step / steps

        c_w = int(600 - 100 * t)
        c_h = int(400 - 100 * t)
        c_x = int(150 - 100 * t)
        c_y = int(60 + 40 * t)
        img_c = ImageTk.PhotoImage(center_img.resize((c_w, c_h)))

        r_w = int(500 + 100 * t)
        r_h = int(300 + 100 * t)
        r_x = int(400 - 250 * t)
        r_y = int(100 - 40 * t)
        img_r = ImageTk.PhotoImage(right_img.resize((r_w, r_h)))

        n_x = int(800 - 400 * t)
        img_n = ImageTk.PhotoImage(next_img.resize((500, 300)))

        app.canvas.create_rectangle(140, 50, 760, 470, fill="#E8DFCA", outline="", tags="carousel")
        app.canvas.create_image(c_x, c_y, anchor="nw", image=img_c, tags="carousel")
        app.canvas.create_image(r_x, r_y, anchor="nw", image=img_r, tags="carousel")
        app.canvas.create_image(n_x, 100, anchor="nw", image=img_n, tags="carousel")

        app.image_refs = {"c": img_c, "r": img_r, "n": img_n}
        app.update()
        app.after(delay)

    app.current_index = (app.current_index + 1) % len(app.images_color)
    draw_carousel(app)
    app.after(1000, lambda: animate_transition(app))

def update_dots(app):
    for i, dot in enumerate(app.dots):
        dot.configure(fg="#212121" if i == app.current_index else "#BDBDBD")