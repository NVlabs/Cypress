# The driving script to generate bookshelf format representation of PCB placement

from pcb import PCBDesign



pcb = PCBDesign(
    project_number='PB310',
    revision='A00',
    destination_folder='./inputs/A00/design/worklib/pb310_a00/',
    # moveable_pages='22-39,41-55,61-75,111-134,140-163'
    # moveable_pages='82-83,89-90,93-94,100-101,111-112,140-141'
    moveable_pages='161-163'
)


pcb.generate_bookshelf(
    output_folder="/home/niansongz/scratch/bookshelf-repo/P161/",
    filter_fixed_on_layer=None, page_range=[161,162,163], tight=True)