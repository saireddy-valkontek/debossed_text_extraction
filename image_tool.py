import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps
import math
import os

# Try to import cairosvg (for SVG handling)
try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False

SUPPORTED_INPUTS = [("All Supported", "*.png *.jpg *.jpeg *.bmp *.webp *.svg"),
                    ("PNG", "*.png"), ("JPG", "*.jpg *.jpeg"), ("SVG", "*.svg")]

def calculate_dpi(width_px, height_px, diagonal_inches):
    pixels_diagonal = math.sqrt(width_px**2 + height_px**2)
    dpi = pixels_diagonal / diagonal_inches
    return round(dpi, 2)

def remove_white_background(image_path, save_path):
    img = Image.open(image_path).convert("RGBA")
    new_data = []
    for item in img.getdata():
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    img.putdata(new_data)
    img.save(save_path, "PNG")

def resize_image(image_path, save_path, new_w, new_h):
    img = Image.open(image_path)
    img = img.resize((new_w, new_h), Image.ANTIALIAS)
    img.save(save_path)

def add_white_bg(image_path, save_path):
    img = Image.open(image_path).convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    bg.save(save_path)

def convert_svg_to_png(svg_path, save_path):
    if CAIROSVG_AVAILABLE:
        cairosvg.svg2png(url=svg_path, write_to=save_path)
        return True
    return False

def convert_svg_to_png_transparent(svg_path, save_path):
    if CAIROSVG_AVAILABLE:
        cairosvg.svg2png(url=svg_path, write_to=save_path, background_color='rgba(0,0,0,0)')
        return True
    return False

class ImageToolApp:
    def __init__(self, master):
        self.master = master
        master.title("🖼️ Image Tool with DPI & Conversion")
        master.geometry("420x420")

        # Image path
        self.img_path = None

        # UI Layout
        tk.Button(master, text="1️⃣ Select Image", command=self.pick_image).grid(row=0, column=0, columnspan=3, pady=5, sticky="we")

        tk.Label(master, text="Width (px):").grid(row=1, column=0)
        self.width_entry = tk.Entry(master)
        self.width_entry.grid(row=1, column=1)

        tk.Label(master, text="Height (px):").grid(row=2, column=0)
        self.height_entry = tk.Entry(master)
        self.height_entry.grid(row=2, column=1)

        tk.Label(master, text="Screen Size (inches):").grid(row=3, column=0)
        self.size_entry = tk.Entry(master)
        self.size_entry.grid(row=3, column=1)

        tk.Button(master, text="🧮 Calculate DPI", command=self.show_dpi).grid(row=3, column=2)
        self.dpi_label = tk.Label(master, text="DPI: -")
        self.dpi_label.grid(row=4, column=0, columnspan=3, pady=5)

        # Options
        row = 5
        tk.Button(master, text="📐 Resize Image", command=self.resize).grid(row=row, column=0, columnspan=3, sticky="we"); row += 1
        tk.Button(master, text="🌈 Make Transparent (remove white)", command=self.transparent).grid(row=row, column=0, columnspan=3, sticky="we"); row += 1
        tk.Button(master, text="🧱 Add White BG to Transparent", command=self.add_bg).grid(row=row, column=0, columnspan=3, sticky="we"); row += 1
        tk.Button(master, text="🔁 SVG to PNG", command=self.svg_to_png).grid(row=row, column=0, columnspan=3, sticky="we"); row += 1
        tk.Button(master, text="🔁 SVG to Transparent PNG", command=self.svg_to_transparent_png).grid(row=row, column=0, columnspan=3, sticky="we"); row += 1

    def pick_image(self):
        self.img_path = filedialog.askopenfilename(filetypes=SUPPORTED_INPUTS)
        if self.img_path:
            messagebox.showinfo("Selected", f"Selected: {os.path.basename(self.img_path)}")

    def show_dpi(self):
        try:
            w = int(self.width_entry.get())
            h = int(self.height_entry.get())
            size = float(self.size_entry.get())
            dpi = calculate_dpi(w, h, size)
            self.dpi_label.config(text=f"DPI: {dpi}")
        except:
            messagebox.showerror("Error", "Please enter valid width, height, and size.")

    def resize(self):
        if not self.img_path:
            messagebox.showerror("Error", "Select an image first.")
            return
        try:
            w = int(self.width_entry.get())
            h = int(self.height_entry.get())
            out = filedialog.asksaveasfilename(defaultextension=".png")
            if out:
                resize_image(self.img_path, out, w, h)
                messagebox.showinfo("Done", "Image resized and saved.")
        except:
            messagebox.showerror("Error", "Invalid width or height.")

    def transparent(self):
        if not self.img_path:
            messagebox.showerror("Error", "Select an image first.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".png")
        if out:
            remove_white_background(self.img_path, out)
            messagebox.showinfo("Done", "White background removed.")

    def add_bg(self):
        if not self.img_path:
            messagebox.showerror("Error", "Select an image first.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".png")
        if out:
            add_white_bg(self.img_path, out)
            messagebox.showinfo("Done", "White background added.")

    def svg_to_png(self):
        if not CAIROSVG_AVAILABLE:
            messagebox.showerror("Missing", "Please install cairosvg:\npip install cairosvg")
            return
        if not self.img_path or not self.img_path.lower().endswith(".svg"):
            messagebox.showerror("Error", "Please select an SVG file.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".png")
        if out:
            convert_svg_to_png(self.img_path, out)
            messagebox.showinfo("Done", "SVG converted to PNG.")

    def svg_to_transparent_png(self):
        if not CAIROSVG_AVAILABLE:
            messagebox.showerror("Missing", "Please install cairosvg:\npip install cairosvg")
            return
        if not self.img_path or not self.img_path.lower().endswith(".svg"):
            messagebox.showerror("Error", "Please select an SVG file.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".png")
        if out:
            convert_svg_to_png_transparent(self.img_path, out)
            messagebox.showinfo("Done", "SVG converted to transparent PNG.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageToolApp(root)
    root.mainloop()