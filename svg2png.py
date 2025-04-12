import cairosvg

def calculate_dpi(resolution_x, resolution_y, display_size_inch):
    # Calculate DPI for horizontal and vertical resolutions
    dpi_x = resolution_x / display_size_inch
    dpi_y = resolution_y / display_size_inch
    # Optionally, return the average or choose either dpi_x or dpi_y
    return max(dpi_x, dpi_y)  # You can use max or avg as per your preference


def svg_to_png(input_svg, output_png, resolution_x, resolution_y, display_size_inch):
    # Calculate the DPI based on resolution and display size
    dpi = calculate_dpi(resolution_x, resolution_y, display_size_inch)

    # Convert SVG to PNG with the calculated DPI and transparent background
    cairosvg.svg2png(url=input_svg, write_to=output_png, dpi=dpi, background_color="transparent")
    print(f"Converted with DPI: {dpi}")


# Example usage
input_svg = r"C:\Users\sai\Downloads\BLUE CLOUD SOFTTECH.svg"  # Path to your input SVG file
output_png = "output_file.png"  # Path to your output PNG file
resolution_x = 240  # Resolution in pixels (width)
resolution_y = 320  # Resolution in pixels (height)
display_size_inch = 2.4  # Display size in inches

svg_to_png(input_svg, output_png, resolution_x, resolution_y, display_size_inch)
