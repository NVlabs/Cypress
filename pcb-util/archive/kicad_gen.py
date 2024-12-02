# The driving script to generate bookshelf format representation of PCB placement

from pcb import PCBDesign



pcb = PCBDesign(
    project_number='PB310',
    revision='A00',
    destination_folder='./inputs/A00/design/worklib/pb310_a00/',
    moveable_pages='82-83,89-90,93-94,100-101,111-112,140-141'
)


# pcb.generate_kicad_board(
#         output_path="./outputs/PB310_kicad/PB310_p154.kicad_pcb", 
#         page_range=list(range(38, 43)))

# pcb.update_placement(pl="/home/scratch.niansongz_research/bookshelf-repo/P140/P140_wl.gp.pl")

pcb.generate_kicad_board(
        output_path="./outputs/PB310_kicad/p140.kicad_pcb", 
        page_range=list(range(140, 151)),
        keep_name=False,
        keep_placement=False)