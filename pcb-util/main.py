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



def generate_benchmark(
    project_name,
    revision,
    worklib,
    page_range,
    load_from_yaml=False,
    visualize=False,
    visualize_pages=False,
    to_bookshelf=False,
    benchmark_name=None,
    target_folder=None,
):
    
    short_name = worklib.split("/")[-1]
    pcb_design = PCB()
    yaml_file = f"./ir/{short_name}.yaml"
    yaml_file_exits = os.path.exists(yaml_file)
    if load_from_yaml and yaml_file_exits:
        with open(yaml_file, "r") as f:
            pcb_dict = yaml.load(f, Loader=yaml.FullLoader)
        pcb_design.from_dict(pcb_dict)
    else:
        pcb_design.import_from_idf(
            project_name=project_name,
            revision=revision,
            worklib=worklib
        )
        # write to yaml
        pcb_dict = pcb_design.to_dict()
        with open(yaml_file, "w") as f:
            yaml.dump(pcb_dict, f)
    

    # visualize
    if visualize:
        if page_range is not None:
            pcb_design.select_pages(page_range)
            # pcb_design.select_layer('TOP')
        pcb_design.visualize(f"./visualize/{benchmark_name}.png", draw_nets=True)
    
    # visualize every page
    if visualize_pages:
        pcb_design.visualize_pages(f"./visualize/{short_name}_pages")

    # convert to bookshelf
    if to_bookshelf:
        # let's mark these pages as movable.
        if page_range is not None:
            pcb_design.select_pages(page_range)
        bs_converter = BookshelfConverter(
            design_name=benchmark_name,
            target_folder=target_folder,
            movable_pages=page_range,
            filter_fixed_on_layer=None
        )

        bs_converter.to_bookshelf(pcb_design)


def process_results(
    pl_path,
    bookshelf_path,
    visualize,
    to_kicad,
    target_kicad_path,
    fig_name=None
):
    # create temporary folder
    name = pl_path.split("/")[-1].split(".")[0]
    temp_folder = f"./intermediate/process_{name}/"
    os.makedirs(temp_folder, exist_ok=True)

    # copy bookshelf files to temporary folder
    os.system(f"cp {bookshelf_path}/* {temp_folder}")
    # remove human placed pl file
    # os.system(f"rm {temp_folder}/*.pl")
    # copy pl file to temporary folder
    # os.system(f"cp {pl_path} {temp_folder}")
    # rename pl file to {name}.pl
    # os.system(f"mv {temp_folder}/{pl_path.split('/')[-1]} {temp_folder}/{name}.pl")

    # get pcb design from bookshelf
    bs_converter = BookshelfConverter(
        design_name=name,
        target_folder=temp_folder,
        filter_fixed_on_layer=None,
        movable_pages=None)
    pcb_design = bs_converter.from_bookshelf()
    
    if visualize:
        if fig_name is None:
            fig_name = name
        pcb_design.visualize(f"./visualize/{fig_name}.pdf", draw_nets=True)
    
    if to_kicad:
        kc_converter = KiCadConverter(target_path=target_kicad_path)
        kc_converter.to_kicad(pcb_design)

def bs_to_kicad():
    # for i in range(1, 11):
    #     bs_path = f"/Cypress/experiments/ns-place-legal/small-{i}"
    #     kc_path = f"/Cypress/experiments/ns-place-routed/small-{i}.kicad_pcb"
    #     # get pcb design from bookshelf
    #     bs_converter = BookshelfConverter(
    #         design_name=f"small-{i}",
    #         target_folder=bs_path,
    #         filter_fixed_on_layer=None,
    #         movable_pages=None)
    #     pcb_design = bs_converter.from_bookshelf()

    #     kc_converter = KiCadConverter(target_path=kc_path)
    #     kc_converter.to_kicad(pcb_design)
    

    bs_path = "/Cypress/experiments/cypress/small-7"
    kc_path = "/Cypress/experiments/cypress/small-7.kicad_pcb"
    # get pcb design from bookshelf
    bs_converter = BookshelfConverter(
        design_name=f"ET095_A00",
        target_folder=bs_path,
        filter_fixed_on_layer=None,
        movable_pages=None)
    pcb_design = bs_converter.from_bookshelf()

    kc_converter = KiCadConverter(target_path=kc_path)
    kc_converter.to_kicad(pcb_design)


def import_from_kicad(
    benchmark_name,
    kicad_path,
    load_from_yaml,
    visualize,
    to_bookshelf,
    bookshelf_path
):
    print("processing ", benchmark_name)
    yaml_file = f"./ir/{benchmark_name}.yaml"
    yaml_file_exits = os.path.exists(yaml_file)
    if load_from_yaml and yaml_file_exits:
        pcb_design = PCB()
        with open(yaml_file, "r") as f:
            pcb_dict = yaml.load(f, Loader=yaml.FullLoader)
        pcb_design.from_dict(pcb_dict)
    else:
        kc_converter = KiCadConverter(target_path=kicad_path)
        pcb_design = kc_converter.from_kicad()
        # write to yaml
        pcb_dict = pcb_design.to_dict()
        with open(yaml_file, "w") as f:
            yaml.dump(pcb_dict, f)

    # translate to origin
    for _, comp in pcb_design.components.items():
        minx = pcb_design.x_range[0]
        miny = pcb_design.y_range[0]
        blx = comp.ctr_x - comp.get_x_length() / 2
        bly = comp.ctr_y - comp.get_y_length() / 2
        newx = blx - minx
        newy = bly - miny
        comp.translate_to(newx, newy)
    
    pcb_design.x_range = [0, pcb_design.x_range[1] - pcb_design.x_range[0]]
    pcb_design.y_range = [0, pcb_design.y_range[1] - pcb_design.y_range[0]]
            
    if visualize:
        pcb_design.visualize(f"./visualize/eval/{benchmark_name}.jpg", draw_nets=True)

    if to_bookshelf:
        bs_converter = BookshelfConverter(
            design_name=benchmark_name,
            target_folder=bookshelf_path,
            filter_fixed_on_layer=None,
            movable_pages=None)
        
        # remove existing files in bookshelf folder
        os.system(f"rm {bookshelf_path}/*")
        bs_converter.to_bookshelf(pcb_design)




def build_dataset():
    with open("info/complete_worklibs.txt", "r") as f:
        worklibs = f.readlines()

    failed = []
    
    for worklib in worklibs:
        proj_name_rev = worklib.strip().split("/")[-1]
        proj_name = proj_name_rev.split("_")[0]
        rev = proj_name_rev.split("_")[1]

        if proj_name_rev + ".yaml" in os.listdir("./ir/"):
            continue

        try:
            generate_benchmark(
                project_name=proj_name,
                revision=rev,
                worklib=worklib.strip(),
                page_range=None,
                load_from_yaml=False,
                visualize=False,
                visualize_pages=False,
                to_bookshelf=False,
                benchmark_name=None,
                target_folder=None,
            )

        except Exception as e:
            failed.append(worklib)
            continue

    with open("info/failed_worklibs.txt", "w") as f:
        for item in failed:
            f.write("%s\n" % item)

def report_completeness(worklib):
    patterns = ["pstxnet.dat", "pstxprt.dat", "page.map", "module_order.dat"]
    idf_patterns = ["*.emn", "*.emp"]
    print("\nChecking worklib: ", worklib)
    for pattern in patterns:
        files = find_file_recursive(worklib, pattern)
        print(files)
        if files is None:
            return False
    for idf in idf_patterns:
        files = find_file_recursive(worklib + "/nv_pcb_export/", idf)
        if files is None:
            return False
    zip_612_file = find_file_recursive(worklib, "612*.zip")
    # check inside zip for a *.csv file
    if zip_612_file is not None:
        thickness_file = find_file_in_zip(zip_612_file, "*pm_thickness.csv")
        if thickness_file is not None:
            pass
        else:
            return False
    else:
        return False

    return True


def generate_small_benchmark():
    from design_configs import small_configs
    for config in small_configs:
        config["to_bookshelf"] = True
        config["target_folder"] = f"/home/niansongz/scratch/pcb-util/outputs/anonymized_small/{config['benchmark_name']}"
        generate_benchmark(**config)



def check_bookshelf():

    from design_configs import small_configs
    for config in small_configs:
        design_name = config["benchmark_name"]
        target_folder = f"/home/niansongz/scratch/pcb-util/outputs/anonymized_small/{config['benchmark_name']}"
        bs_converter = BookshelfConverter(
            design_name=design_name,
            target_folder=target_folder,
            filter_fixed_on_layer=None,
            movable_pages=None)
        pcb_design = bs_converter.from_bookshelf()
        pcb_design.visualize(f"./visualize/{design_name}.png", draw_nets=True)


def pick_new_benchmark():
    from design_configs import small20_config
    generate_benchmark(**small20_config)


def generate_mesh_dataset():
    # this is a crazy thought
    # I use the pin positions (x, y, side) as the vertices
    # All nets and its endpoints are converted to a face
    # then I turn these info to .obj format 3D mesh data.
    # The issue with using manifold diffusion model is that the input has 
    # to be a mesh, which has spatial placement info already
    # i don't know, maybe add some noise as input
    
    dataset_folder = "./ir/dataset"
    # get all yaml file full path under dataset_folder
    yaml_files = [os.path.join(root, file) for root, _, files in os.walk(dataset_folder) 
                  for file in files if file.endswith('.yaml')]
    
    pcb_designs = []
    # pcb_manifold dataset folder
    results_folder = "./outputs/pcb_manifold/"

    # Create the results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)

    for idx, yaml_file in enumerate(yaml_files):
        with open(yaml_file, 'r') as f:
            pcb_dict = yaml.safe_load(f)
        pcb_design = PCB()
        pcb_design.from_dict(pcb_dict)
        
        obj_filename = os.path.join(results_folder, f"pcb_{idx}.obj")
        
        with open(obj_filename, 'w') as obj_file:
            # Dictionary to keep track of vertex indices
            vertex_indices = {}
            vertex_count = 1

            # Write vertices
            for comp_name, component in pcb_design.components.items():
                for pin_name, pin in component.pins.items():
                    z = 1 if component.layer == "TOP" else 0
                    obj_file.write(f"v {pin.abs_x} {pin.abs_y} {z}\n")
                    vertex_indices[(comp_name, pin_name)] = vertex_count
                    vertex_count += 1

            # Write faces (nets)
            for net_name, net in pcb_design.nets.items():
                if len(net.pins) >= 3:
                    for i in range(len(net.pins) - 2):
                        v1 = vertex_indices[tuple(net.pins[0])]
                        v2 = vertex_indices[tuple(net.pins[i+1])]
                        v3 = vertex_indices[tuple(net.pins[i+2])]
                        obj_file.write(f"f {v1} {v2} {v3}\n")

        print(f"Generated {obj_filename}")

    print(f"Generated .obj files for {len(yaml_files)} PCB designs in {results_folder}")

    print(f"Generated .obj files for {len(pcb_designs)} PCB designs in {results_folder}")


if __name__ == "__main__":
    # generate_mesh_dataset()
    # generate_small_benchmark()
    # check_bookshelf()
    bs_to_kicad()
    # build_dataset()
    # res = report_completeness("/home/niansongz/scratch/syseng/Projects/E3631/A00/design/worklib/e3631_a00")
    # print(res)

    # pick_new_benchmark()
