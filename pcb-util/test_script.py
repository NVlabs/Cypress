import yaml
import os
from bookshelf_converter import BookshelfConverter
from kicad_converter import KiCadConverter
from pcb import PCB


bookshelf_path = "/home/niansongz/scratch/pcb-util/benchmark/big-1/results/cypress/bookshelf"


bs_converter = BookshelfConverter(
    design_name="big-1",
    target_folder=bookshelf_path,
    filter_fixed_on_layer=None,
    movable_pages=None,
)

pcb = bs_converter.from_bookshelf()
pcb.visualize("visualize/big-1-cypress.jpg")
