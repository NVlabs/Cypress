import yaml
import os
from bookshelf_converter import BookshelfConverter
from kicad_converter import KiCadConverter
from pcb import PCB
from bs_utils import find_file_recursive, find_file_in_zip

from design_configs import *

# to be able to load tuple from yaml
def tuple_constructor(loader, node):
    # Construct the Python tuple from a YAML sequence node
    return tuple(loader.construct_sequence(node))

# Add the constructor to the SafeLoader
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)



bs_converter = BookshelfConverter(
    design_name="small-7",
    target_folder="/home/nz264/shared/Cypress/benchmarks/small-7",
    filter_fixed_on_layer=None,
    movable_pages=None)
pcb_design = bs_converter.from_bookshelf()
pcb_design.visualize(f"./visualize/small-7.pdf", draw_nets=True)
kc_converter = KiCadConverter(target_path="kicad/small-7.kicad_pcb")
kc_converter.to_kicad(pcb_design)