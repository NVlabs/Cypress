import os
import json
from pcb import PCBDesign, find_file_recursive

# first let's test if we can read newer/older kicad files
from kiutils.board import Board
from kiutils.items.fpitems import FpText, FpPoly

pcb = Board().from_file("/home/scratch.niansongz_research/pcb-benchmark/small-1/results/ns-place/routed.kicad_pcb")
# ok, so newer version works, older version doesn't have footprint info


bench_root_path = '/home/scratch.niansongz_research/pcb-benchmark'


def find_size_of_fp(fp):
    # find the fpPoly in graphic items
    poly = None
    for gi in fp.graphicItems:
        if isinstance(gi, FpPoly):
            poly = gi
            break

    assert poly is not None, "cannot find FpPoly in graphic items"

    # find the bounding box
    minx = float('inf')
    miny = float('inf')
    maxx = float('-inf')
    maxy = float('-inf')
    for coord in poly.coordinates:
        minx = min(minx, coord.X)
        miny = min(miny, coord.Y)
        maxx = max(maxx, coord.X)
        maxy = max(maxy, coord.Y)
    
    return maxx - minx, maxy - miny


# for kicad, we generate bookshelf format (pl file) to measure HPWL, net crossing, etc.
# apply to sa-pcb, ns-place, quilter
def generate_bookshelf_from_kicad():
    def convert_one(input_kicad7_path, routed_kicad_path, input_bookshelf_path, refdes_anon_table):
        assert os.path.exists(input_kicad7_path), f"input kicad7 file {input_kicad7_path} does not exist"
        # assert os.path.exists(routed_kicad_path), f"routed kicad file {routed_kicad_path} does not exist"
        if not os.path.exists(routed_kicad_path):
            # print in red
            print(f"\033[91mWarning: routed kicad file {routed_kicad_path} does not exist\033[0m")
            return
        input_pcb = Board().from_file(input_kicad7_path)
        routed_pcb = Board().from_file(routed_kicad_path)
        # build a map: routed_pcb footprint name -> input_pcb footprint name
        routed_to_input_name_table = {}
        # then, we will use the refdes_anon_table to map input_pcb footprint name to refdes
        assert len(input_pcb.footprints) == len(routed_pcb.footprints), "footprint count mismatch"
        for input_fp, routed_fp in zip(input_pcb.footprints, routed_pcb.footprints):
            # if not 'reference' in routed_fp.properties:
                # import ipdb; ipdb.set_trace()
            routed_name = routed_fp.properties["Reference"]
            # we have to find the name from graph items
            input_name = None
            for gi in input_fp.graphicItems:
                if isinstance(gi, FpText):
                    if gi.type == "reference":
                        input_name = gi.text
                        break
            assert input_name is not None, "cannot find reference in input footprint"
            routed_to_input_name_table[routed_name] = input_name

        # read refdes anon table (a json file)
        with open(refdes_anon_table, "r") as f:
            refdes_anon = json.load(f)

        routed_kicad_parent = os.path.dirname(routed_kicad_path)
        target_bs_path = os.path.join(routed_kicad_parent, "bookshelf")
        # os.system(f"rm -r {target_bs_path}")
        # return
        # copy input_bookshelf_path to target_bs_path
        os.system(f"cp -r {input_bookshelf_path} {target_bs_path}")
        
        pl_path = find_file_recursive(target_bs_path, "*.pl")

        # find min, max x, y
        minx = float('inf')
        miny = float('inf')
        maxx = float('-inf')
        maxy = float('-inf')
        for routed_fp in routed_pcb.footprints:
            pos = routed_fp.position
            minx = min(minx, pos.X)
            miny = min(miny, pos.Y)
            maxx = max(maxx, pos.X)
            maxy = max(maxy, pos.Y)
        
        # replace the pl file
        f = open(pl_path, "w")
        # write header
        f.write("UCLA pl 1.0\n")
        for routed_fp in routed_pcb.footprints:
            name = routed_fp.properties["Reference"]
            # find the input name
            input_name = routed_to_input_name_table[name]
            # find the refdes
            refdes = refdes_anon[input_name]
            # find the position
            pos = routed_fp.position
            orient = None
            if pos.angle == 0:
                orient = "N"
            elif pos.angle == 90 or pos.angle == -270:
                orient = "E"
            elif pos.angle == 180 or pos.angle == -180:
                orient = "S"
            elif pos.angle == 270 or pos.angle == -90:
                orient = "W"
            else:
                orient = "N"
            center_x = pos.X
            center_y = pos.Y
            width, height = find_size_of_fp(routed_fp)
            bl_x = center_x - width / 2
            bl_y = center_y - height / 2
            bl_x -= minx
            bl_y -= miny
            f.write(f"{refdes} {bl_x * 10} {bl_y * 10} : {orient}\n")

    for tool in ['sa-pcb', 'ns-place', 'quilter']:
        for bench in range(1, 11):
            benchname = f"small-{bench}"
            res_path = os.path.join(bench_root_path, benchname, "results", tool)
            input_bookshelf_path = os.path.join(bench_root_path, benchname, "bookshelf")
            refdes_anon_table = os.path.join(bench_root_path, benchname, "kicad", "refdes_anon_table.json")
            input_kicad7_path = os.path.join(bench_root_path, benchname, "kicad", f"{benchname}_kicad7.kicad_pcb")
            if tool == "quilter":
                routed_kicad_path = os.path.join(res_path, f"{benchname}_kicad4.kicad_pcb")
            else:
                routed_kicad_path = os.path.join(res_path, "routed.kicad_pcb")
            convert_one(input_kicad7_path, routed_kicad_path, input_bookshelf_path, refdes_anon_table)


# for bookshelf pl, we generate kicad format to evaluate routability
def generate_kicad_from_pl():
    
    def generate_one(project_number, revision, worklib, moveable_pages, pl_path, kicad_path, page_range):
        pcb = PCBDesign(
            project_number=project_number,
            revision=revision,
            destination_folder=worklib,
            moveable_pages=moveable_pages
        )
        pcb.update_placement(pl=pl_path)
        pcb.generate_kicad_board(
            output_path=kicad_path,
            page_range=page_range,
            keep_name=True,
            keep_placement=True
        )
    
    project_numbers = ['PB310'] * 3 + ["PB201"] * 3 + ["ET095"] * 2 + ["PN42C"] * 2
    worklibs = ['./inputs/XDR/Crocodile/PB310/A00/design/worklib/pb310_a00'] * 3 + \
        ['/home/niansongz/scratch/pcb-util/inputs/PB201/A00/design/worklib/pb201_a00'] * 3 + \
        ['/home/niansongz/scratch/pcb-util/inputs/XDR/BM/ET095/A00/design/worklib/et095_a00'] * 2 + \
        ['/home/niansongz/scratch/pcb-util/inputs/Hippo/Boards/SwitchBoard/PN42C/A00/design/worklib/pn42c_a00'] * 2
    movable_pages = ['39', '100-190', '100-190', '100-190', '43', '73', '87', '32', '66-69', '30-36', '115-117']
    page_ranges = [[39], list(range(140, 152)), [161, 162, 163], [43], [73], [87], [32], [66, 67, 68, 69], [30, 31, 32, 33, 34, 35, 36], [115, 116, 117]]

    for i in range(1, 11):
    # for i in [2]:
        benchname = f"small-{i}"
        result_path = os.path.join(bench_root_path, benchname, "results", "cypress-sep11")
        # find the pl file
        pl_path = find_file_recursive(result_path, "*.pl")
        assert pl_path is not None, f"cannot find pl file in {result_path}"
        output_kicad = os.path.join(result_path, 'routed.kicad_pcb')
        generate_one(project_numbers[i-1], 'A00', worklibs[i-1], movable_pages[i-1], pl_path, output_kicad, page_ranges[i-1])



if __name__ == "__main__":
    # generate_bookshelf_from_kicad()
    generate_kicad_from_pl()