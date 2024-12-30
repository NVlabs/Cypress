import yaml
import os
from bookshelf_converter import BookshelfConverter
from kicad_converter import KiCadConverter
from nsplace_converter import KiCad4Converter
from pcb import PCB
from bs_utils import find_file_recursive, find_file_in_zip

from design_configs import *

# to be able to load tuple from yaml
def tuple_constructor(loader, node):
    # Construct the Python tuple from a YAML sequence node
    return tuple(loader.construct_sequence(node))

# Add the constructor to the SafeLoader
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)



# bs_converter = BookshelfConverter(
#     design_name="small-7",
#     target_folder="/home/nz264/shared/Cypress/benchmarks/small-7",
#     filter_fixed_on_layer=None,
#     movable_pages=None)
# pcb_design = bs_converter.from_bookshelf()
# pcb_design.visualize(f"./visualize/small-7.pdf", draw_nets=True)
# kc_converter = KiCadConverter(target_path="kicad/small-7.kicad_pcb")
# kc_converter.to_kicad(pcb_design)


ns_place_kicad = "/home/nz264/shared/Cypress/experiments/ns-place"
for i in range(1, 11, 1): #1-10
    kicad_file = f"{ns_place_kicad}/output.small-{i}_kicad4.kicad_pcb"
    converter = KiCad4Converter(kicad_file)
    pcb = converter.from_kicad()
    pcb.visualize(f"{ns_place_kicad}/small-{i}_kicad4.png", draw_nets=True)
    bs_converter = BookshelfConverter(
        design_name=f"small-{i}",
        target_folder=f"/home/nz264/shared/Cypress/experiments/ns-place-bs/small-{i}",
        filter_fixed_on_layer=None,
        movable_pages=[0]
    )
    bs_converter.to_bookshelf(pcb)