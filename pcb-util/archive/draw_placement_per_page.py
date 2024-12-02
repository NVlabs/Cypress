# Driving script to draw placement per page
from pcb import PCBDesign

pcb = PCBDesign(
    project_number='PB310',
    revision='A00',
    destination_folder="/home/niansongz/scratch/pcb-util/inputs/XDR/Crocodile/PB310/A00/design/worklib/pb310_a00",
    moveable_pages=''
)

pcb.draw_placement_per_page(
    output_folder="./outputs/per_page/pb310"
)

# pcb.report_height_distribution(
#     bin_size=1,
#     out_file="./outputs/height_distribution.png"
# )