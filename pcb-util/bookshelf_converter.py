import os
import datetime
import math
import bs_utils

from pcb import Pin, Net, Component, PCB


class BookshelfConverter:
    def __init__(
        self, 
        design_name,
        target_folder,
        filter_fixed_on_layer,
        movable_pages,
        unit_factor=10.0,
        anonymous=True
    ):
        self.design_name = design_name
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        self.target_folder = target_folder

        # four options: None, "TOP", "BOTTOM", "BOTH"
        self.filter_fixed_on_layer = filter_fixed_on_layer
        self.unit_factor = unit_factor

        # movable pages
        if isinstance(movable_pages, str):
            movable_pages = bs_utils.get_pages(movable_pages)
        elif movable_pages is None:
            movable_pages = []
        self.movable_pages_list = movable_pages

        self.nodes_file = os.path.join(self.target_folder, f"{design_name}.nodes")
        self.nets_file = os.path.join(self.target_folder, f"{design_name}.nets")
        self.pl_file = os.path.join(self.target_folder, f"{design_name}.pl")
        self.scl_file = os.path.join(self.target_folder, f"{design_name}.scl")
        self.aux_file = os.path.join(self.target_folder, f"{design_name}.aux")

        self.anonymous = anonymous

    def to_bookshelf(self, pcb):
        # produce bookshelf files
        self.pcb = pcb
        self.generate_nodes_file()
        self.generate_nets_file()
        self.generate_pl_file()
        self.generate_scl_file()
        self.generate_aux_file()
        print("Written Bookshelf files to ", self.target_folder)

    def from_bookshelf(self):
        # create a brand new PCB object from bookshelf files
        self.pcb = PCB()
        self.pcb.name = self.design_name
        self.read_nodes_file()
        self.read_nets_file()
        self.read_pl_file()
        return self.pcb


    def generate_nodes_file(self):
        # nodes file defines the name and the size of rectangular components
        header_lines = f"UCLA nodes 1.0\n\n"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
        second_line = f"# {self.design_name}.nodes generated on {timestamp} using {os.path.abspath(__file__)} {1.0}\n"
        nodes_data = [header_lines, second_line]

        num_terminals = 0
        num_nodes = 0
        
        for cname, comp in self.pcb.components.items():
            num_nodes += 1
            comp_name = comp.anon_name if self.anonymous else comp.name 
            if comp.page in self.movable_pages_list:
                # movable component
                movetype = ""
            else:
                # fixed component
                movetype = "terminal"
                num_terminals += 1
                # If it is fixed, we decide if we want to filter it out
                if self.filter_fixed_on_layer and (
                    self.filter_fixed_on_layer == comp.layer
                    or self.filter_fixed_on_layer == "BOTH"
                ):
                    continue

            x_length = int(comp.get_x_length() * self.unit_factor)
            y_length = int(comp.get_y_length() * self.unit_factor)
            
            node_line = f"{comp_name} {x_length} {y_length} {movetype}"
            nodes_data.append(node_line)

        nodes_data[0] += f"NumNodes : {num_nodes}\nNumTerminals : {num_terminals}\n"
        bs_utils.write_file(self.nodes_file, nodes_data)

    def read_nodes_file(self):
        with open(self.nodes_file, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            # skip header and comments:
            skip_patterns = ["UCLA", "NumNodes", "NumTerminals", "#"]
            if any([pattern in line for pattern in skip_patterns]):
                continue
            # skip empty lines
            if not line.strip():
                continue
            # parse the line
            items = line.split()
            # depending on the number of items
            if len(items) == 3:
                # movable component
                comp_name, x_length, y_length = items
            elif len(items) == 4:
                # fixed component
                comp_name, x_length, y_length, movetype = items
            x_length = float(x_length) / self.unit_factor
            y_length = float(y_length) / self.unit_factor
            # create a component object rectangle centered at zero
            rectangle = [
                [-x_length / 2, -y_length / 2],
                [x_length / 2, -y_length / 2],
                [x_length / 2, y_length / 2],
                [-x_length / 2, y_length / 2]
            ]
            new_comp = Component(comp_name, None, rectangle, 0, 0, None, None, None)

            # add to PCB object
            self.pcb.components[comp_name] = new_comp
            

    def generate_nets_file(self):
        header_lines = f"UCLA nets 1.0\n\n"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
        second_line = f"# {self.design_name}.nets generated on {timestamp} using {os.path.abspath(__file__)} {1.0}\n"
        nets_data = [header_lines, second_line]

        total_num_pins = 0
        num_nets = 0

        for net_name, net in self.pcb.nets.items():
            if len(net.pins) == 0:
                continue
            # skip GND, GROUND nets
            if "GND" in net_name or "GROUND" in net_name:
                continue
            # if net name starts with digit, we prepend it with "N"
            if net_name[0].isdigit():
                net_name = "N" + net_name
            net_name = net_name.replace("*", "") # remove any star

            if self.anonymous:
                net_name = net.anon_name
            net_line = f"NetDegree : {len(net.pins)} {net_name}"
            nets_data.append(net_line)

            total_num_pins += len(net.pins)
            for pin in net.pins:
                comp_name, pin_name = pin
                # see if we anonymize the component name
                if self.anonymous:
                    comp_anon_name = self.pcb.components[comp_name].anon_name
                    pin__anon_name = self.pcb.components[comp_name].pins[pin_name].anon_name
                pin_obj = self.pcb.components[comp_name].pins[pin_name]
                rltv_x = int(pin_obj.rltv_x * self.unit_factor)
                rltv_y = int(pin_obj.rltv_y * self.unit_factor)
                if self.anonymous:
                    pin_line = f"\t{comp_anon_name} I : {rltv_x} {rltv_y}"
                else:
                    pin_line = f"\t{comp_name} I : {rltv_x} {rltv_y}"
                nets_data.append(pin_line)

            num_nets += 1   

        nets_data[0] += f"NumNets : {num_nets}\nNumPins : {total_num_pins}\n"
        bs_utils.write_file(self.nets_file, nets_data)

    
    def read_nets_file(self):
        with open(self.nets_file, "r") as f:
            lines = f.readlines()
        
        line_idx = 0
        while line_idx < len(lines):
            # skip header and comments:
            skip_patterns = ["UCLA", "NumNets", "NumPins", "#"]
            line = lines[line_idx]
            if any([pattern in line for pattern in skip_patterns]):
                line_idx += 1
                continue
            # skip empty lines
            if not line.strip():
                line_idx += 1
                continue
            if "NetDegree" in line:
                net_name = line.split()[-1]
                net_degree = int(line.split()[-2])
                net_pins = []
                new_net = Net(net_name, net_name)
            for i in range(net_degree):
                line_idx += 1
                pin_line = lines[line_idx]
                # example pin_line: "U59 I : 13 -25"
                comp_name, _, _, rltv_x, rltv_y = pin_line.split()
                rltv_x = float(rltv_x) / self.unit_factor
                rltv_y = float(rltv_y) / self.unit_factor
                pin_name = str(len(self.pcb.components[comp_name].pins))
                new_pin = Pin(pin_name, pin_name, comp_name, net_name, rltv_x, rltv_y, None, None)
                # pin goes to component
                self.pcb.components[comp_name].pins[pin_name] = new_pin
                # net records the name of the component and pin
                new_net.pins.append([comp_name, pin_name])
            line_idx += 1 # move to the next net
            # add net to PCB object
            self.pcb.nets[net_name] = new_net


    def generate_pl_file(self):
        header_lines = f"UCLA pl 1.0\n\n"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
        second_line = f"# {self.design_name}.pl generated on {timestamp} using {os.path.abspath(__file__)} {1.0}\n"
        pl_data = [header_lines, second_line]

        for cname, comp in self.pcb.components.items():
            ctr_x = comp.ctr_x
            ctr_y = comp.ctr_y
            x_length = comp.get_x_length()
            y_length = comp.get_y_length()
            bl_x = ctr_x - x_length / 2
            bl_y = ctr_y - y_length / 2
            # bookshelf doesn't support negative coordinates
            # so we minus the minx and miny of the canvas
            bl_x -= self.pcb.x_range[0]
            bl_y -= self.pcb.y_range[0]
            bl_x = int(bl_x * self.unit_factor)
            bl_y = int(bl_y * self.unit_factor)
            fixed = "/FIXED" if comp.page not in self.movable_pages_list else ""
            
            # an alternative way that marks all small components as movable
            # fixed = "FIXED"
            # if comp.layer == "TOP" and x_length * y_length < 3:
            #     fixed = ""
            orient = "N" if comp.layer == "TOP" else "FN"
            comp_name = comp.anon_name if self.anonymous else cname
            pl_data.append(f"{comp_name} {bl_x} {bl_y} : {orient} {fixed}")
        
        bs_utils.write_file(self.pl_file, pl_data)

    def read_pl_file(self):
        # four jobs: translate component, rotate it, and assign layer, set canvas size
        with open(self.pl_file, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            # skip header and comments:
            skip_patterns = ["UCLA", "#"]
            if any([pattern in line for pattern in skip_patterns]):
                continue
            # skip empty lines
            if not line.strip():
                continue
            # parse the line
            items = line.split()
            if len(items) == 5:
                # movable component
                comp_name, bl_x, bl_y, _, orient = items
                self.pcb.components[comp_name].moveable = True
            else:
                # fixed component
                comp_name, bl_x, bl_y, _, orient, _ = items
                self.pcb.components[comp_name].moveable = False
            
            # assign layer
            if 'F' in orient:
                self.pcb.components[comp_name].layer = "BOTTOM"
            else:
                self.pcb.components[comp_name].layer = "TOP"
            
            bl_x = float(bl_x) / self.unit_factor
            bl_y = float(bl_y) / self.unit_factor
            x_length = self.pcb.components[comp_name].get_x_length()
            y_length = self.pcb.components[comp_name].get_y_length()
            ctr_x = bl_x + x_length / 2
            ctr_y = bl_y + y_length / 2
            self.pcb.components[comp_name].translate_to(ctr_x, ctr_y)
            if "N" in orient:
                theta = 0
            elif "W" in orient:
                theta = 90
            elif "S" in orient:
                theta = 180
            elif "E" in orient:
                theta = 270
            # convert to radians
            theta = math.radians(theta)
            self.pcb.components[comp_name].rotate(theta)
        
        # set canvas size
        minx = float('inf')
        miny = float('inf')
        maxx = float('-inf')
        maxy = float('-inf')
        for cname, comp in self.pcb.components.items():
            minx = min(minx, comp.ctr_x - comp.get_x_length() / 2)
            miny = min(miny, comp.ctr_y - comp.get_y_length() / 2)
            maxx = max(maxx, comp.ctr_x + comp.get_x_length() / 2)
            maxy = max(maxy, comp.ctr_y + comp.get_y_length() / 2)
        self.pcb.x_range = [minx, maxx]
        self.pcb.y_range = [miny, maxy]

    def generate_scl_file(self):
        # scl file defines the canvas size
        header_lines = f"UCLA scl 1.0\n\n"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
        second_line = f"# {self.design_name}.scl generated on {timestamp} using {os.path.abspath(__file__)} {1.0}\n"
        scl_data = [header_lines, second_line]

        canvas_width = int((self.pcb.x_range[1] - self.pcb.x_range[0]) * self.unit_factor)
        canvas_height = int((self.pcb.y_range[1] - self.pcb.y_range[0]) * self.unit_factor)

        site_width, row_height = 1, 1
        num_rows = int(canvas_height / row_height)
        num_sites = int(canvas_width / site_width)
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

        bs_utils.write_file(self.scl_file, scl_data)
    
    def generate_aux_file(self):
        aux_data = [
            f"RowBasedPlacement : {self.design_name}.nodes {self.design_name}.nets {self.design_name}.pl {self.design_name}.scl"
        ]
        bs_utils.write_file(self.aux_file, aux_data)
