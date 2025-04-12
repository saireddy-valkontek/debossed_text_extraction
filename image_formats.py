import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import math
import os

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False

def calculate_dpi(res_x, res_y, size_inches):
    diagonal_pixels = math.sqrt(res_x**2 + res_y**2)
    dpi = diagonal_pixels / size_inches
    return round(dpi, 2)

def add_background(image_path, save_path):
    img = Image.open(image_path).convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])  # Use alpha channel as mask
    bg.save(save_path, "PNG")

def make_png_transparent(image_path, save_path):
    img = Image.open(image_path).convert("RGBA")
    datas = img.getdata()
    new_data = []

    for item in datas:
        # Detect white-ish backgrounds and make them transparent
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            new_data.append((255, 255, 255, 0))  # Transparent
        else:
            new_data.append(item)

    img.putdata(new_data)
    img.save(save_path, "PNG")

def convert_svg_to_png(svg_path, png_path, transparent=True):
    if CAIROSVG_AVAILABLE:
        cairosvg.svg2png(url=svg_path, write_to=png_path)
        return True
    return False

def convert_png_to_svg(png_path, svg_path):
    if CAIROSVG_AVAILABLE:
        cairosvg.png2svg(url=png_path, write_to=svg_path)
        return True
    return False

class ImageToolApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Tool + DPI + Conversion")
        master.geometry("360x340")

        # DPI Section
        tk.Label(master, text="Resolution (Width x Height)").grid(row=0, column=0)
        self.res_x = tk.Entry(master, width=10)
        self.res_x.grid(row=0, column=1)
        self.res_y = tk.Entry(master, width=10)
        self.res_y.grid(row=0, column=2)

        tk.Label(master, text="Screen Size (inches)").grid(row=1, column=0)
        self.screen_size = tk.Entry(master, width=10)
        self.screen_size.grid(row=1, column=1)

        tk.Button(master, text="Calculate DPI", command=self.get_dpi).grid(row=1, column=2)

        self.dpi_result = tk.Label(master, text="")
        self.dpi_result.grid(row=2, column=0, columnspan=3)

        # Conversion Buttons
        row = 3
        tk.Button(master, text="Add White BG to PNG", command=self.add_bg).grid(row=row, column=0, columnspan=3, sticky="we"); row += 1
        tk.Button(master, text="Convert SVG to PNG (Transparent)", command=self.svg_to_png).grid(row=row, column=0, columnspan=3, sticky="we"); row += 1
        tk.Button(master, text="Convert PNG to SVG", command=self.png_to_svg).grid(row=row, column=0, columnspan=3, sticky="we"); row += 1
        tk.Button(master, text="Make PNG Transparent", command=self.png_to_transparent).grid(row=row, column=0, columnspan=3, sticky="we")

    def get_dpi(self):
        try:
            x = int(self.res_x.get())
            y = int(self.res_y.get())
            size = float(self.screen_size.get())
            dpi = calculate_dpi(x, y, size)
            self.dpi_result.config(text=f"DPI: {dpi}")
        except:
            messagebox.showerror("Error", "Enter valid resolution and size.")

    def add_bg(self):
        file_path = filedialog.askopenfilename(filetypes=[("PNG Files", "*.png")])
        if file_path:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
            add_background(file_path, save_path)
            messagebox.showinfo("Done", "White background added to PNG.")

    def svg_to_png(self):
        if not CAIROSVG_AVAILABLE:
            messagebox.showerror("Missing Module", "Install cairosvg: pip install cairosvg")
            return
        file_path = filedialog.askopenfilename(filetypes=[("SVG Files", "*.svg")])
        if file_path:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
            success = convert_svg_to_png(file_path, save_path)
            if success:
                messagebox.showinfo("Done", "SVG converted to PNG.")
            else:
                messagebox.showerror("Error", "Conversion failed.")

    def png_to_svg(self):
        if not CAIROSVG_AVAILABLE:
            messagebox.showerror("Missing Module", "Install cairosvg: pip install cairosvg")
            return
        file_path = filedialog.askopenfilename(filetypes=[("PNG Files", "*.png")])
        if file_path:
            save_path = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG Files", "*.svg")])
            success = convert_png_to_svg(file_path, save_path)
            if success:
                messagebox.showinfo("Done", "PNG converted to SVG.")
            else:
                messagebox.showerror("Error", "Conversion failed.")

    def png_to_transparent(self):
        file_path = filedialog.askopenfilename(filetypes=[("PNG Files", "*.png")])
        if file_path:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
            make_png_transparent(file_path, save_path)
            messagebox.showinfo("Done", "White background removed from PNG.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageToolApp(root)
    root.mainloop()