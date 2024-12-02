import os

from pcb import PCBDesign
from archive.convert_kicad_version import convert_kicad_version

def generate_benchmark(
    benchmark_name,
    project_number,
    revision,
    worklib,
    moveable_pages,
    benchmark_repo_path,
    page_range,
    bs_filter_fixed_on_layer,
    bs_tight_canvas,
    kc_keep_name,
    kc_keep_pl,
):
    # three formats: 
    # Bookshelf, KiCAD 7, KiCAD 4
    pcb = PCBDesign(
        project_number=project_number,
        revision=revision,
        destination_folder=worklib,
        moveable_pages=moveable_pages
    )

    # create a folder for the benchmark
    bookshelf_folder = os.path.join(benchmark_repo_path, benchmark_name, "bookshelf")
    if not os.path.exists(bookshelf_folder):
        os.makedirs(bookshelf_folder)
    pcb.generate_bookshelf(
        output_folder=bookshelf_folder,
        filter_fixed_on_layer=bs_filter_fixed_on_layer,
        page_range=page_range,
        tight=bs_tight_canvas
    )

    print("generated bookshelf: ", bookshelf_folder)

    # create a folder for kicad files
    kicad_folder = os.path.join(benchmark_repo_path, benchmark_name, "kicad")
    if not os.path.exists(kicad_folder):
        os.makedirs(kicad_folder)

    kicad7_filename = os.path.join(kicad_folder, f"{benchmark_name}_kicad7.kicad_pcb")
    kicad4_filename = os.path.join(kicad_folder, f"{benchmark_name}_kicad4.kicad_pcb")

    pcb.generate_kicad_board(
        output_path=kicad7_filename,
        page_range=page_range,
        keep_name=kc_keep_name,
        keep_placement=kc_keep_pl
    )

    print("generated kicad7: ", kicad7_filename)

    convert_kicad_version(kicad7_filename, kicad4_filename)

    print("generated kicad4: ", kicad4_filename)


def generate_all_small_benchmarks():
    generate_benchmark(
        benchmark_name="small-1",
        project_number='PB310',
        revision='A00',
        worklib='./inputs/XDR/Crocodile/PB310/A00/design/worklib/pb310_a00',
        moveable_pages='39',
        benchmark_repo_path="/home/scratch.niansongz_research/pcb-benchmark",
        page_range=[39],
        bs_filter_fixed_on_layer=None,
        bs_tight_canvas=True,
        kc_keep_name=False,
        kc_keep_pl=True
    )

    generate_benchmark(
        benchmark_name="small-2",
        project_number='PB310',
        revision='A00',
        worklib='./inputs/XDR/Crocodile/PB310/A00/design/worklib/pb310_a00',
        moveable_pages='100-190',
        benchmark_repo_path="/home/scratch.niansongz_research/pcb-benchmark",
        page_range=list(range(140, 152)),
        bs_filter_fixed_on_layer=None,
        bs_tight_canvas=True,
        kc_keep_name=False,
        kc_keep_pl=True
    )

    generate_benchmark(
        benchmark_name="small-3",
        project_number='PB310',
        revision='A00',
        worklib='./inputs/XDR/Crocodile/PB310/A00/design/worklib/pb310_a00',
        moveable_pages='100-190',
        benchmark_repo_path="/home/scratch.niansongz_research/pcb-benchmark",
        page_range=[161, 162, 163],
        bs_filter_fixed_on_layer=None,
        bs_tight_canvas=True,
        kc_keep_name=False,
        kc_keep_pl=True
    )

    generate_benchmark(
        benchmark_name="small-4",
        project_number='PB201',
        revision='A00',
        worklib='/home/niansongz/scratch/pcb-util/inputs/PB201/A00/design/worklib/pb201_a00',
        moveable_pages='43',
        benchmark_repo_path="/home/scratch.niansongz_research/pcb-benchmark",
        page_range=[43],
        bs_filter_fixed_on_layer=None,
        bs_tight_canvas=True,
        kc_keep_name=False,
        kc_keep_pl=True
    )

    generate_benchmark(
        benchmark_name="small-5",
        project_number='PB201',
        revision='A00',
        worklib='/home/niansongz/scratch/pcb-util/inputs/PB201/A00/design/worklib/pb201_a00',
        moveable_pages='73',
        benchmark_repo_path="/home/scratch.niansongz_research/pcb-benchmark",
        page_range=[73],
        bs_filter_fixed_on_layer=None,
        bs_tight_canvas=True,
        kc_keep_name=False,
        kc_keep_pl=True
    )

    generate_benchmark(
        benchmark_name="small-6",
        project_number='PB201',
        revision='A00',
        worklib='/home/niansongz/scratch/pcb-util/inputs/PB201/A00/design/worklib/pb201_a00',
        moveable_pages='87',
        benchmark_repo_path="/home/scratch.niansongz_research/pcb-benchmark",
        page_range=[87],
        bs_filter_fixed_on_layer=None,
        bs_tight_canvas=True,
        kc_keep_name=False,
        kc_keep_pl=True
    )

    generate_benchmark(
        benchmark_name="small-7",
        project_number='ET095',
        revision='A00',
        worklib='/home/niansongz/scratch/pcb-util/inputs/XDR/BM/ET095/A00/design/worklib/et095_a00',
        moveable_pages='32',
        benchmark_repo_path="/home/scratch.niansongz_research/pcb-benchmark",
        page_range=[32],
        bs_filter_fixed_on_layer=None,
        bs_tight_canvas=True,
        kc_keep_name=False,
        kc_keep_pl=True
    )

    generate_benchmark(
        benchmark_name="small-8",
        project_number='ET095',
        revision='A00',
        worklib='/home/niansongz/scratch/pcb-util/inputs/XDR/BM/ET095/A00/design/worklib/et095_a00',
        moveable_pages='66-69',
        benchmark_repo_path="/home/scratch.niansongz_research/pcb-benchmark",
        page_range=[66, 67, 68, 69],
        bs_filter_fixed_on_layer=None,
        bs_tight_canvas=True,
        kc_keep_name=False,
        kc_keep_pl=True
    )

    generate_benchmark(
        benchmark_name="small-9",
        project_number='PN42C',
        revision='A00',
        worklib='/home/niansongz/scratch/pcb-util/inputs/Hippo/Boards/SwitchBoard/PN42C/A00/design/worklib/pn42c_a00',
        moveable_pages='30-36',
        benchmark_repo_path="/home/scratch.niansongz_research/pcb-benchmark",
        page_range=list(range(30, 37)),
        bs_filter_fixed_on_layer=None,
        bs_tight_canvas=True,
        kc_keep_name=False,
        kc_keep_pl=True
    )

    generate_benchmark( # CPLD
        benchmark_name="small-10",
        project_number='PN42C',
        revision='A00',
        worklib='/home/niansongz/scratch/pcb-util/inputs/Hippo/Boards/SwitchBoard/PN42C/A00/design/worklib/pn42c_a00',
        moveable_pages='115-117',
        benchmark_repo_path="/home/scratch.niansongz_research/pcb-benchmark",
        page_range=list(range(115, 118)),
        bs_filter_fixed_on_layer=None,
        bs_tight_canvas=True,
        kc_keep_name=False,
        kc_keep_pl=True
    )

def generate_all_big_benchmarks():
    generate_benchmark(
        benchmark_name="big-1",
        project_number='PB310',
        revision='A00',
        worklib='./inputs/XDR/Crocodile/PB310/A00/design/worklib/pb310_a00',
        moveable_pages='22-39,41-55,102-105',
        benchmark_repo_path="/home/scratch.niansongz_research/pcb-benchmark",
        page_range=None, # all pages
        bs_filter_fixed_on_layer=None,
        bs_tight_canvas=False,
        kc_keep_name=False,
        kc_keep_pl=True
    )

if __name__ == "__main__":
    generate_all_small_benchmarks()