import cairosvg
from lxml import etree

def remove_background_and_convert(input_svg: str, output_png: str):
    # Parse the SVG file
    tree = etree.parse(input_svg)
    root = tree.getroot()

    # Remove any background elements (e.g., <rect> with a solid fill)
    for element in root.findall(".//{http://www.w3.org/2000/svg}rect"):
        if 'fill' in element.attrib and element.attrib['fill'] != 'none':
            # Remove the solid background rect element
            root.remove(element)

    # Save the modified SVG (without the background)
    modified_svg = "modified_no_background.svg"
    tree.write(modified_svg)

    # Convert the modified SVG to PNG with transparency
    cairosvg.svg2png(url=modified_svg, write_to=output_png)

    print(f"SVG background removed and converted to transparent PNG: {output_png}")

# Path to input SVG and output PNG file
input_svg =r"C:\Users\sai\Downloads\BLUE CLOUD SOFTTECH.svg"
output_png = "output_transparent.png"

# Remove background and convert
remove_background_and_convert(input_svg, output_png)
