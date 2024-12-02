# Driving script to export placed PCB designs to IDF format
from pcb import PCBDesign


placed = "/home/scratch.niansongz_research/bookshelf-repo/PB310_A00/placed/PB310_A00.gp.pl"

pcb = PCBDesign(
    project_number='PB310',
    revision='A00',
    destination_folder='./inputs/A00/design/worklib/pb310_a00/',
    moveable_pages='1-2,4'
)

pcb.generate_allegro_placement(
    pl=placed,
    output_folder="./outputs/PCB310_A00_placed/")