# A class to represent, import, export, and update a PCB design
import os
import datetime
import sys
import zipfile
import csv
import shutil
import fnmatch
import json
import bs_utils


class PCBDesign:
    def __init__(
        self,
        project_number,
        revision,
        destination_folder,
        moveable_pages,
        unit_factor=10.0,
    ):
        self.project_number = project_number
        self.revision = revision
        self.design_name = f"{project_number}_{revision}"
        self.unit_factor = unit_factor

        self.fixed_pages_list = []
        self.moveable_pages_list = get_pages(moveable_pages) if moveable_pages else []

        # worklib e.g. PB201/A00/design/worklib/pb201_a00
        self.destination_folder = destination_folder

        self.prepare_files()

        self.page_per_ref_dict, self.refdeses_per_page_dict = bs_utils.main(
            pstxprt=self.pstxprt,
            page_map=self.page_map,
            module_order=self.module_order,
            project_number="",
            revision="",
            destination_folder=self.destination_folder,
        )

        self.pstxnet_net_dict, self.pstxnet_refdes_dict = parse_net(self.pstxnet)

        self.parsed_idf = IDF(self.emn_file, self.emp_file)
        self.components = [
            comp
            for comp in self.parsed_idf.components
            if comp.refdes not in ["NOREFDES", ""]
        ]

        self.component_dict = {}
        for comp in self.components:
            # get page number
            # If REFDES appears on more than one page, use the smallest page number
            page_number = self.page_per_ref_dict[comp.refdes]["pdf_page_number"][0]
            comp.page_number = page_number
            self.component_dict[comp.refdes] = comp

        self.pm_thickness_dict = self.parse_pm_thickness_csv_file(
            self.pm_thickness_csv_file
        )
        self.refdes_dict = {}
        self.min_x = float("inf")
        self.min_y = float("inf")
        self.max_x = 0
        self.max_y = 0

    def prepare_files(self):
        interm_folder = os.path.join(
            os.path.dirname(__file__), "intermediate", self.design_name
        )
        if not os.path.exists(interm_folder):
            os.makedirs(interm_folder)
            print("Intermediate folder created: " + interm_folder)
        # find auxillary files
        names = ["pstxnet.dat", "pstxprt.dat", "page.map", "module_order.dat"]
        for name in names:
            if find_file_recursive(interm_folder, name):
                continue # skip if already exists
            file = find_file_recursive(self.destination_folder, name)
            if file:
                shutil.copy(file, interm_folder)
            else:
                raise ValueError(f"File {name} not found in {self.destination_folder}")

        self.pstxnet = os.path.join(interm_folder, "pstxnet.dat")
        self.pstxprt = os.path.join(interm_folder, "pstxprt.dat")
        self.page_map = os.path.join(interm_folder, "page.map")
        self.module_order = os.path.join(interm_folder, "module_order.dat")

        # find IDF files
        idf_patterns = ["*.emn", "*.emp"]
        for idf in idf_patterns:
            if find_file_recursive(interm_folder, idf):
                continue # skip if already exists
            file = find_file_recursive(
                os.path.join(self.destination_folder, "nv_pcb_export"), idf
            )
            if file:
                shutil.copy(file, interm_folder)
            else:
                raise ValueError(
                    f"IDF file {idf} not found in {self.destination_folder}"
                )

        self.emn_file = find_file_recursive(interm_folder, "*.emn")
        self.emp_file = find_file_recursive(interm_folder, "*.emp")

        # find thickness file

        # already exists?
        previously_copied = find_file_recursive(interm_folder, "*pm_thickness.csv")
        if previously_copied:
            self.pm_thickness_csv_file = previously_copied
            return

        zip_612_file = find_file_recursive(self.destination_folder, "612*.zip")
        # check inside zip for a *.csv file
        if zip_612_file is not None:
            thickness_file = find_file_in_zip(zip_612_file, "*pm_thickness.csv")
            if thickness_file is not None:
                shutil.copy(zip_612_file, interm_folder)
            new_zip_612_file = os.path.join(
                interm_folder, os.path.basename(zip_612_file)
            )
            # unzip it
            with zipfile.ZipFile(new_zip_612_file, "r") as zip_ref:
                zip_ref.extractall(interm_folder)
        else:
            raise ValueError("Thickness file not found.")

        self.pm_thickness_csv_file = find_file_recursive(
            interm_folder, "*pm_thickness.csv"
        )

    def generate_nodes_file(self, output_file):
        """
        data for file PCB.nodes : describes each node (component REFDES) and its bounding box size and MOVABLE attribute
                1) File starts with some header lines.
                2) Second line will be a file description comment line starting with "#"
                    <filename> generated on <TIMESTAMP> using <CODEPATHNAME> <CODE REVISION>
                3) For .nodes, the format is: node_name width height movetype
                    node_name - the REFDES name
                    width - 2D rectangle width of the REFDES in nominal position (as defined as the X-range in LAYOUT EMP file) - aka known as “vertical and face up” – N (North)
                    height - 2D rectangle height of the REFDES in nominal position (as defined as the Y-range in LAYOUT EMP file) - do not confuse this param with the Z-height value in EMP
                    movetype - If movetype is not specified, then it's assumed movable, otherwise it is either a terminal node (fixed node), or terminal_NI node (fixed node, but overlap is allowed with this node)
                4) Data source for PCB.node
                    Layout EMN and EMP files
                        get PACKAGE_NAME (JEDEC TYPE) per REFDES in .PLACEMENT section
                        get bounding box in N position from the X and Y ranges in the EMP file for that PACKAGE_NAME (JEDEC TYPE)
        """
        header_lines = f"UCLA nodes 1.0\n\n"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
        second_line = f"# {self.design_name}.nodes generated on {timestamp} using {os.path.abspath(__file__)} {1.0}\n"
        nodes_data = [header_lines, second_line]

        num_terminals = 0
        num_nodes = 0

        for c, comp in enumerate(self.components):
            # if comp.layer != self.target_layer: continue
            page_number = self.page_per_ref_dict[comp.refdes]["pdf_page_number"][
                0
            ]  # If REFDES appears on more than one page, use the smallest page number

            if self.page_range is not None:
                if int(page_number) not in self.page_range:
                    continue

            node_name = comp.refdes

            if self.moveable_pages_list:
                movetype = "" if page_number in self.moveable_pages_list else "terminal"
            else:
                movetype = "terminal" if page_number in self.fixed_pages_list else ""

            if movetype:  # if fixed
                if self.filter_fixed_on_layer and (
                    self.filter_fixed_on_layer == comp.layer
                    or self.filter_fixed_on_layer == "BOTH"
                ):
                    # print(node_name, comp.layer, movetype)
                    continue
                num_terminals += 1

            num_nodes += 1

            width = int(
                (comp.shape.bounds[2] - comp.shape.bounds[0])
                * self.unit_factor
            )
            height = int(
                (comp.shape.bounds[3] - comp.shape.bounds[1])
                * self.unit_factor
            )

            node_line = f"{node_name} {width} {height} {movetype}"
            nodes_data.append(node_line)
            self.refdes_dict[comp.refdes] = {
                "page_number": page_number,
                "movetype": movetype,
                "layer": comp.layer,
                "angle": comp.angle,
                "left_lower_x": int(comp.original_shape.bounds[0] * self.unit_factor), # relative to center
                "left_lower_y": int(comp.original_shape.bounds[1] * self.unit_factor), # relative to center
            }

        nodes_data[0] += f"NumNodes : {num_nodes}\nNumTerminals : {num_terminals}\n"
        write_file(output_file, nodes_data)

    def generate_nets_file(self, output_file):
        """
        data for PCB.nets : Describes the design netlist
            1) File starts with some header lines - see example file
            2) Second line will be a file description comment line starting with "#"
                <filename> generated on <TIMESTAMP> using <CODEPATHNAME> <CODE REVISION>
            3) For each NETNAME
                "NetDegree :" <N - number of pins connected to NETNAME> <NETNAME>
                For each of the connected N pins:
                    <PINNAME> I : <X> <Y>
                    where X,Y are the pin center X,Y offsets from the center of the REFDES in nominal position
                    It does not matter to have the correct IO direction - use "I" for al pins
            4) Data source for PCB.nets:
                1) PCB_pm_thickness.csv file inside released_files/612 zip. e.g. from 612-1B310-1000-A00.zip :
                    1) COMP_PACKAGE,REFDES,SYM_CENTER_X,SYM_CENTER_Y,PIN_NUMBER,PIN_X,PIN_Y,SYM_MIRROR,PASTE_MASK_HEIGHT
                    2) CON_QSFP_038_SMT_RA_F_P080,J2,417.265,266.750,40B,421.963,258.350,YES,0.156
                    3) CON_QSFP_038_SMT_RA_F_P080,J2,417.265,266.750,40A,419.237,258.350,YES,0.156
                    4) CAP_SMD_7343,C3956,242.800,303.400,1,242.800,306.512,YES,0.127
                    5) CAP_SMD_7343,C3956,242.800,303.400,2,242.800,300.288,YES,0.127
                    6) BGA_4784_P100_070X070,U1_2,322.575,165.500,XAC9,328.075,163.768,NO,0.127
                2) TOP side = SYM_MIRROR=="NO"; BOTTOM side = SYM_MIRROR=="YES";
                3) Note that pm_thickness.csv file provides PIN's X,Y in actual position which may be different from the required nominal position
                    1) If REFDES is on TOP (SYM_MIRROR=="NO") and rotation_angle = 0, then it is in nominal position
                    2) Otherwise, it needs to be rotated and/or moved to TOP side, and adjust X,Y accordingly via method - TBD.
                    3) This conversion back to nominal is not supported for the initial MPV
        """
        header_lines = f"UCLA nets 1.0\n\n"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
        second_line = f"# {self.design_name}.nets generated on {timestamp} using {os.path.abspath(__file__)} {1.0}\n"
        nets_data = [header_lines, second_line]

        # prepare nets_data
        total_num_pins = 0
        num_nets = 0
        count_skipped = 0
        for netname in self.pstxnet_net_dict:
            pins_num_per_net = 0
            netname_fixed = netname.replace("'", "")

            pin_data = []

            for ref_pin in self.pstxnet_net_dict[netname]:
                refdes, pin_name = ref_pin.split("-")

                if refdes in self.refdes_dict:  # filtering
                    # page_number = self.refdes_dict[refdes]["page_number"]
                    # SYM_CENTER_X,SYM_CENTER_Y are the overall REFDES center X,Y
                    self.refdes_dict[refdes].update(
                        {
                            "center_x": self.pm_thickness_dict[refdes][0]["center_x"],
                            "center_y": self.pm_thickness_dict[refdes][0]["center_y"],
                        }
                    )

                    for pin_dict in self.pm_thickness_dict[refdes]:
                        if pin_name == pin_dict["pin_number"]:
                            x = pin_dict["pin_x"] - pin_dict["center_x"]
                            y = pin_dict["pin_y"] - pin_dict["center_y"]
                            # pin_line = f'\t{refdes}_PG{page_number}-{pin_name} I : {x} {y}'
                            # pin_line = f"\t{refdes}_PG{page_number} I : {x} {y}"
                            pin_line = f"\t{refdes} I : {x} {y}"
                            pins_num_per_net += 1
                            pin_data.append(pin_line)
                            break
                # else:
                #     count_skipped +=1
                #     if not refdes.startswith('VS'):
                #         print(netname, refdes)

            # skip over nets with bad names but update centex in refdes_dict
            if (
                "GND" in netname_fixed.upper()
                or "GROUND" in netname_fixed.upper()
                # or netname_fixed[0].isdigit()
            ):
                # print(netname_fixed)
                continue

            if netname_fixed[0].isdigit():
                # bookshelf doesn't allow netname starting with digit
                netname_fixed = f"NET_{netname_fixed}"

            total_num_pins += pins_num_per_net

            if pins_num_per_net:
                netdegree_line = f"NetDegree : {pins_num_per_net} {netname_fixed}"
                nets_data.append(netdegree_line)
                num_nets += 1
                nets_data += pin_data
            # if num_nets > 10: break

        # print(count_skipped)
        nets_data[0] += f"NumNets : {num_nets}\nNumPins : {total_num_pins}\n"

        write_file(output_file, nets_data)

    def generate_aux_file(self, output_file):
        aux_data = [
            f"RowBasedPlacement : {self.design_name}.nodes {self.design_name}.nets {self.design_name}.pl {self.design_name}.scl"
        ]
        write_file(output_file, aux_data)

    def parse_pm_thickness_csv_file(self, pm_thickness_csv_file):
        """
            file content:
                Allegro Report
                C:/Perforce/Projects/XDR/Crocodile/PB310/A00/design/worklib/pb310_a00/physical/142-1B310-1000-A00.brd
                2023/10/18 22:49:46

                COMP_PACKAGE,REFDES,SYM_CENTER_X,SYM_CENTER_Y,PIN_NUMBER,PIN_X,PIN_Y,SYM_MIRROR,PASTE_MASK_HEIGHT
                CON_QSFP_038_SMT_RA_F_P080,J2,417.265,266.750,40B,421.963,258.350,YES,0.156
                CON_QSFP_038_SMT_RA_F_P080,J2,417.265,266.750,40A,419.237,258.350,YES,0.156

        For each NETNAME
                "NetDegree :" <N - number of pins connected to NETNAME> <NETNAME>
                For each of the connected N pins:
                    <PINNAME> I : <X> <Y>
                    where X,Y are the pin center X,Y offsets from the center of the REFDES in nominal position
                    It does not matter to have the correct IO direction - use "I" for al pins
        Data source for PCB.nets:
                TOP side = SYM_MIRROR=="NO"; BOTTOM side = SYM_MIRROR=="YES";
                Note that pm_thickness.csv file provides PIN's X,Y in actual position which may be different from the required nominal position
                    1) If REFDES is on TOP (SYM_MIRROR=="NO") and rotation_angle = 0, then it is in nominal position
                    2) Otherwise, it needs to be rotated and/or moved to TOP side, and adjust X,Y accordingly via method - TBD.
                    3) This conversion back to nominal is not supported for the initial MPV
        """
        pm_thickness_dict = {}
        with open(pm_thickness_csv_file, "r") as f:
            csvreader = csv.reader(f)

            for c, row in enumerate(csvreader):
                if c < 5 or not row:
                    continue
                refdes = row[1]
                center_x = int(float(row[2]) * self.unit_factor)
                center_y = int(float(row[3]) * self.unit_factor)
                pin_number = row[4]
                pin_x = int(float(row[5]) * self.unit_factor)
                pin_y = int(float(row[6]) * self.unit_factor)
                layer = "TOP" if row[-2] == "NO" else "BOTTOM"
                pin_dict = {
                    "pin_number": pin_number,
                    "layer": layer,
                    "center_x": center_x,
                    "center_y": center_y,
                    "pin_x": pin_x,
                    "pin_y": pin_y,
                }
                if refdes in pm_thickness_dict:
                    pm_thickness_dict[refdes].append(pin_dict)
                else:
                    pm_thickness_dict[refdes] = [pin_dict]
                # if c>60: break

        # print(pm_thickness_dict)
        return pm_thickness_dict

    def generate_pl_file(self, output_file):
        """
        data PCB.pl : Gives the coordinates (x,y) and orientation for each REFDES
            The coordinates for all movable REFDES will be (0,0) or undefined
            For .pl, the format is: node_name lowerleft_Xcoordinate lowerleft_Ycoordinate : orientation movetype
            node_name - the REFDES name
            lowerleft_Xcoordinate - X location of the FIXED REFDES center; "0" if movable
            lowerleft_Ycoordinate - Y location of the FIXED REFDES center; "0" if movable
            : - a semicolon
            orientation - any value of: N or E or S or W (compass direction values)
            For FIXED REFDES : N (if PLACEMENT ROTATION angle=0); E (angle=90); S (angle=180); W (angle=270);
            For MOVEABLE REFDES: N
            movetype - For movetype, must be /FIXED if terminal and /FIXED_NI if terminal_NI
        """
        header_lines = f"UCLA pl 1.0\n\n"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
        second_line = f"# {self.design_name}.pl generated on {timestamp} using {os.path.abspath(__file__)} {1.0}\n"
        pl_data = [header_lines, second_line]

        for refdes in self.refdes_dict:
            movetype = "/FIXED" if self.refdes_dict[refdes]["movetype"] else ""
            # page_number = self.refdes_dict[refdes]["page_number"]
            node_name = refdes
            lowerleft_Xcoordinate = (
                self.refdes_dict[refdes]["center_x"]
                + self.refdes_dict[refdes]["left_lower_x"]
            )
            lowerleft_Ycoordinate = (
                self.refdes_dict[refdes]["center_y"]
                + self.refdes_dict[refdes]["left_lower_y"]
            )

            upperright_Xcoordinate = (
                self.refdes_dict[refdes]["center_x"]
                - self.refdes_dict[refdes]["left_lower_x"]
            )
            upperright_Ycoordinate = (
                self.refdes_dict[refdes]["center_y"]
                - self.refdes_dict[refdes]["left_lower_y"]
            )

            self.min_x = min(self.min_x, lowerleft_Xcoordinate)
            self.min_y = min(self.min_y, lowerleft_Ycoordinate)
            self.max_x = max(self.max_x, upperright_Xcoordinate)
            self.max_y = max(self.max_y, upperright_Ycoordinate)

            angle = self.refdes_dict[refdes]["angle"]
            if angle == 0:
                orientation = "N"
            elif angle == 90:
                orientation = "E"
            elif angle == 180:
                orientation = "S"
            elif angle == 270:
                orientation = "W"
            else: # other angles are not supported, use N instead
                orientation = "N"
            refdes_line = f"{node_name} {lowerleft_Xcoordinate} {lowerleft_Ycoordinate} : {orientation} {movetype}"
            pl_data.append(refdes_line)

        write_file(output_file, pl_data)

    def generate_scl_data(self, output_file):
        """
        data for PCB.scl : circuit row information : Specifies the placement image (individual circuit rows for standard-cell placement)
                    In digital VLSI design, all components are snapped on a placement grid (with sites and rows). This definition helps to avoid component overlap after placement.
                    Defining the site width and row height as minimal distance between the components.
                    The file contains text blocks for each of the rows. See reference info for file format
        """
        header_lines = f"UCLA scl 1.0\n\n"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
        second_line = f"# {self.design_name}.scl generated on {timestamp} using {os.path.abspath(__file__)} {1.0}\n"
        scl_data = [header_lines, second_line]

        # prepare scl data per anthony's pcb.py code
        # DB units: 1 = 0.1mm
        # fp_width = 425 * 10  # mm
        # fp_height = 325 * 10  # mm

        if self.bookself_tight_layout:
            fp_width = self.max_x - self.min_x
            fp_width *= 1.2
            fp_height = self.max_y - self.min_y
            fp_height *= 1.2
        else:
            (
                self.min_x,
                self.min_y,
                self.max_x,
                self.max_y,
            ) = self.parsed_idf.board_outline.bounds
            fp_width = int(self.max_x - self.min_x) * self.unit_factor
            fp_height = int(self.max_y - self.min_y) * self.unit_factor

        min_dist = 1  # mm
        # estimate floorplan
        site_width, row_height = min_dist, min_dist
        # width = closest_div(width, site_width)
        # height = closest_div(height, row_height)
        num_rows = int(fp_height // row_height)
        num_sites = int(fp_width // site_width)
        scl_data.append(f"NumRows : {num_rows}\n\n")
        for i in range(num_rows):
            orient = "1" if (i % 2) == 0 else "0"
            content = """CoreRow Horizontal
    Coordinate   : %d
    Height       : %d
    Sitewidth    : %d
    Sitespacing  : %d
    Siteorient   : %s
    Sitesymmetry : 1
    SubrowOrigin : 0  NumSites : %d
End
    """ % (
                i * row_height,
                row_height,
                site_width,
                site_width,
                orient,
                num_sites,
            )
            scl_data.append(content)

        write_file(output_file, scl_data)

    def generate_bookshelf(
        self,
        output_folder,
        filter_fixed_on_layer=None,
        page_range=None,
        unit_factor=10,
        tight=False,
    ):
        """
        generate bookshelf files
        """
        # create output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.bookself_tight_layout = tight

        # filter_fixed_on_layer choices: ['TOP', 'BOTTOM', 'BOTH', None]
        self.filter_fixed_on_layer = filter_fixed_on_layer
        self.page_range = page_range
        self.generate_nodes_file(f"{output_folder}/{self.design_name}.nodes")
        self.generate_nets_file(f"{output_folder}/{self.design_name}.nets")
        self.generate_aux_file(f"{output_folder}/{self.design_name}.aux")
        self.generate_pl_file(f"{output_folder}/{self.design_name}.pl")
        self.generate_scl_data(f"{output_folder}/{self.design_name}.scl")

    def generate_placed_idf(self, pl, output_folder):
        """
        generate placed IDF files
        """
        # create output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # copy emp files
        shutil.copy(self.emp_file, f"{output_folder}/{self.design_name}.emp")

        # read pl file, build dict: {refdes: [x, y, rotation]}
        placement = {}
        with open(pl, "r") as f:
            pl_lines = f.readlines()
        for line in pl_lines[2:]:
            # example line: R27_1 645 1445 : W
            line = line.strip().split()
            if len(line) != 5:
                continue
            refdes = line[0]
            x = float(line[1]) / self.unit_factor
            y = float(line[2]) / self.unit_factor
            orientation = line[4]
            if orientation == "N":
                rotation = 0
            elif orientation == "E":
                rotation = 90
            elif orientation == "S":
                rotation = 180
            elif orientation == "W":
                rotation = 270
            elif rotation == "FN":
                rotation = 0
            elif rotation == "FE":
                rotation = 90
            elif rotation == "FS":
                rotation = 180
            elif rotation == "FW":
                rotation = 270
            else:
                rotation = 0

            placement[refdes] = [x, y, rotation]

        # read emn files as lines
        with open(self.emn_file, "r") as f:
            emn_lines = f.readlines()
        new_emn_lines = []
        line_idx = 0

        # before .PLACEMENT
        while line_idx < len(emn_lines):
            line = emn_lines[line_idx]
            # remove the final newline character
            line = line.replace("\n", "")
            new_emn_lines.append(line)
            line_idx += 1
            if line.strip() == ".PLACEMENT":
                break  # start of placement section

        while line_idx < len(emn_lines):
            line = emn_lines[line_idx]
            # packagename, part number, refdes
            line = line.strip().split()
            if len(line) != 3:
                break
            _, _, refdes = line
            next_line = emn_lines[line_idx + 1]
            next_line = next_line.strip().split()
            if len(next_line) != 6:
                raise ValueError(f"Error in emn file: {next_line}")
            x, y, offset, angle, side, status = next_line
            # check if refdes is in placement dict
            if refdes in placement:
                x, y, angle = placement[refdes]
            new_emn_lines.append(emn_lines[line_idx].replace("\n", ""))
            new_emn_lines.append(
                f"\t{refdes}\t{x}\t{y}\t{offset}\t{angle}\t{side}\t{status}"
            )
            line_idx += 2

        # after .PLACEMENT
        while line_idx < len(emn_lines):
            line = emn_lines[line_idx]
            new_emn_lines.append(line)
            line_idx += 1

        write_file(f"{output_folder}/{self.design_name}.emn", new_emn_lines)

    def generate_allegro_placement(self, pl, output_folder):
        # create output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # read pl file, build dict: {refdes: [x, y, rotation]}
        placement = {}
        with open(pl, "r") as f:
            pl_lines = f.readlines()
        for line in pl_lines[2:]:
            # example line: R27_1 645 1445 : W
            line = line.strip().split()
            if len(line) != 5:
                continue
            refdes = line[0]
            x = float(line[1]) / self.unit_factor
            y = float(line[2]) / self.unit_factor
            orientation = line[4]
            if orientation == "N":
                rotation = 0
            elif orientation == "E":
                rotation = 90
            elif orientation == "S":
                rotation = 180
            elif orientation == "W":
                rotation = 270
            elif rotation == "FN":
                rotation = 0
            elif rotation == "FE":
                rotation = 90
            elif rotation == "FS":
                rotation = 180
            elif rotation == "FW":
                rotation = 270
            else:
                rotation = 0

            placement[refdes] = [x, y, rotation]

        allegro_placement = ["UUNITS = MM"]
        for refdes in placement:
            x, y, rotation = placement[refdes]
            placement_string = f"{refdes} {x} {y} {rotation}"
            if refdes in self.component_dict:
                comp = self.component_dict[refdes]
                if comp.layer == "BOTTOM":
                    placement_string += " m"
            allegro_placement.append(placement_string)

        write_file(f"{output_folder}/placement.txt", allegro_placement)

    def draw_placement_per_page(self, output_folder):
        # first make sure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # maintain a dict of page number to polygons
        page_polygons = {}
        for comp in self.components:
            page_number = self.page_per_ref_dict[comp.refdes]["pdf_page_number"][0]
            if page_number not in page_polygons:
                page_polygons[page_number] = []
            page_polygons[page_number].append(comp.shape)

        # the function to draw polygons
        import matplotlib.pyplot as plt
        from shapely.geometry import Polygon
        from matplotlib.patches import Polygon as MplPolygon

        def draw_polygons(polygons, filename):
            """
            Draws a list of polygons using matplotlib.

            Parameters:
            polygons (list): List of shapely.geometry.Polygon objects
            filename (str): The name of the output file
            """
            fig, ax = plt.subplots()
            minx = float("inf")
            miny = float("inf")
            maxx = 0
            maxy = 0

            for poly in polygons:
                if isinstance(poly, Polygon):
                    mpl_poly = MplPolygon(
                        list(poly.exterior.coords),
                        closed=True,
                        edgecolor="black",
                        facecolor="lightblue",
                    )
                    ax.add_patch(mpl_poly)
                    minx = min(minx, poly.bounds[0])
                    miny = min(miny, poly.bounds[1])
                    maxx = max(maxx, poly.bounds[2])
                    maxy = max(maxy, poly.bounds[3])
                else:
                    print("The list contains a non-Polygon object.")

            # make sure content is displayed
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            # save to file
            fig.savefig(filename)

        for page_number in page_polygons:
            draw_polygons(
                page_polygons[page_number], f"{output_folder}/page_{page_number}.png"
            )

    def report_height_distribution(self, bin_size, out_file):
        """
        Draws a histogram from a list of heights and saves it to an image file.

        Parameters:
        heights (list): List of float numbers representing heights
        bin_size (int): The size of the bins for the histogram
        output_path (str): Path to save the output image
        """
        import matplotlib.pyplot as plt

        heights = [float(comp.height) for comp in self.components]

        # Calculate the number of bins
        min_height = min(heights)
        max_height = max(heights)
        bins = range(int(min_height), int(max_height) + bin_size * 2, bin_size)

        # Create the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(heights, bins=bins, edgecolor="black")
        # write the frequency on top of each bar
        for i in range(len(bins) - 1):
            freq = int(plt.hist(heights, bins=bins, edgecolor="black")[0][i])
            if freq == 0:
                continue
            plt.text(
                (bins[i] + bins[i + 1]) / 2,
                plt.hist(heights, bins=bins, edgecolor="black", facecolor="#76B900")[0][
                    i
                ],
                str(freq),
                ha="center",
                va="bottom",
            )

        # use log y axis
        plt.yscale("log")

        # Add titles and labels
        fontsize = 18
        plt.title("Height Distribution", fontsize=fontsize)
        plt.xlabel("Height (mm)", fontsize=fontsize)
        plt.ylabel("Frequency", fontsize=fontsize)

        # Save the histogram to the specified path
        plt.savefig(out_file)
        plt.close()

    def generate_kicad_board(
        self, output_path, page_range=None, keep_name=True, keep_placement=True
    ):
        import kiutils
        from kiutils.board import Board
        from kiutils.footprint import Footprint, Pad
        from kiutils.items.common import Position, Net, PageSettings
        from kiutils.items.fpitems import FpPoly

        # get the folder name of output filename (output_path)
        output_folder = os.path.dirname(output_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # if not keep name, we need two lookup tables
        # new net name -> original net name
        # new refdes -> original refdes
        net_name_table = {}
        refdes_table = {}

        board = Board().create_new()

        # page width, height
        bl_x, bl_y, tr_x, tr_y = self.parsed_idf.board_outline.bounds
        page_width = tr_x - bl_x
        page_height = tr_y - bl_y
        board.paper = PageSettings(
            paperSize="User", width=int(page_width), height=int(page_height)
        )

        # build list of nets
        # find pin-to-net mapping
        pin_to_net = {}
        for idx, netname in enumerate(self.pstxnet_net_dict):
            if "GND" in netname.upper() or "GROUND" in netname.upper():
                continue
            # build a kicad net object
            net = Net(idx, netname if keep_name else f"NET_{idx}")
            if not keep_name:
                net_name_table[f'NET_{idx}'] = netname
            board.nets.append(net)  # add to board
            for ref_pin in self.pstxnet_net_dict[netname]:
                # ref_pin refers to "REFDES-PIN" format pin name string
                pin_to_net[ref_pin] = net

        for idx, comp in enumerate(self.components):
            if page_range is not None:
                if int(comp.page_number) not in page_range:
                    continue

            name = comp.refdes if keep_name else f"COMP_{idx}"
            if not keep_name:
                refdes_table[f'COMP_{idx}'] = comp.refdes
            footprint = Footprint().create_new(
                library_id=comp.part_number,
                value=name,
                reference=name,
            )
            # set layer
            if comp.layer == "TOP":
                footprint.layer = "F.Cu"
            else:
                footprint.layer = "B.Cu"
            # set position
            cx, cy = comp.position
            footprint.position = Position(cx, cy)
            if not keep_placement:
                footprint.position = Position(0, 0)

            # graphic shape polygon
            coords = []
            for x, y in comp.shape.exterior.coords:
                coords.append(Position(x - cx, y - cy))  # relative position
            fp_layer = "F.SilkS" if comp.layer == "TOP" else "B.SilkS"
            fp_poly = FpPoly(coordinates=coords, fill="solid", layer=fp_layer)
            footprint.graphicItems.append(fp_poly)
            board.footprints.append(footprint)

            # reduce problem size: only add pins for moveable pages
            # if not comp.page_number in self.moveable_pages_list:
            #     continue

            # add pins
            for pin_dict in self.pm_thickness_dict[comp.refdes]:
                pin_x = pin_dict["pin_x"] - pin_dict["center_x"]
                pin_y = pin_dict["pin_y"] - pin_dict["center_y"]
                pin_x /= self.unit_factor
                pin_y /= self.unit_factor
                pad_position = Position(pin_x, pin_y)
                pin_name = pin_dict["pin_number"]
                layer = "F.Cu" if comp.layer == "TOP" else "B.Cu"
                ref_pin = f"{comp.refdes}-{pin_name}"
                pad = Pad(
                    number=pin_name,
                    position=pad_position,
                    layers=[layer],
                    size=Position(0.1, 0.1),
                )
                if ref_pin in pin_to_net:
                    net = pin_to_net[ref_pin]  # find the kicad net token
                    pad.net = net
                    footprint.pads.append(pad)

        board.to_file(output_path)

        if not keep_name:
            # write two tables to output_folder
            # dump as json, indent=4
            with open(f"{output_folder}/net_name_anon_table.json", "w") as f:
                json.dump(net_name_table, f, indent=4)
            with open(f"{output_folder}/refdes_anon_table.json", "w") as f:
                json.dump(refdes_table, f, indent=4)

    def update_placement(self, pl):
        import shapely

        # read pl file, build dict: {refdes: [x, y, rotation]}
        placement = {}
        with open(pl, "r") as f:
            pl_lines = f.readlines()
        for line in pl_lines[2:]:
            # example line: R27_1 645 1445 : W
            line = line.strip().split()
            if len(line) != 5:
                continue
            refdes = line[0]
            x = float(line[1]) / self.unit_factor
            y = float(line[2]) / self.unit_factor
            orientation = line[4]
            if orientation == "N":
                rotation = 0
            elif orientation == "W":
                rotation = 90
            elif orientation == "S":
                rotation = 180
            elif orientation == "E":
                rotation = 270
            elif orientation == "FN":
                rotation = 0
            elif orientation == "FW":
                rotation = 90
            elif orientation == "FS":
                rotation = 180
            elif orientation == "FE":
                rotation = 270
            else:
                rotation = 0

            placement[refdes] = [x, y, rotation]

        for comp in self.components:
            if comp.refdes in placement:
                x, y, rotation = placement[comp.refdes]
                # since bookshelf pl uses bottom left corner as the reference point
                # we need to convert that into center point
                min_x, min_y, max_x, max_y = comp.shape.bounds
                x = x + (max_x - min_x) / 2
                y = y + (max_y - min_y) / 2
                # move comp.shape which is a shapely polygon
                # to the new position
                xoff = x - comp.position[0]
                yoff = y - comp.position[1]
                comp.shape = shapely.affinity.translate(
                    comp.original_shape, xoff=xoff, yoff=yoff
                )
                # comp.shape = shapely.affinity.translate(
                #     comp.original_shape, xoff=x, yoff=y
                # )
                comp.shape = shapely.affinity.rotate(comp.shape, rotation)
                comp.position = (x, y)
