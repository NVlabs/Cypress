# an intermediate representation for a PCB design
import os
import math
import bs_utils
import yaml


class Pin:
    def __init__(
        self,
        pin_name,
        anon_name,
        component,
        net,
        rltv_x,
        rltv_y,
        abs_x,
        abs_y
    ):
        # should be easy to dump and load
        self.pin_name = pin_name
        self.anon_name = anon_name
        self.component = component # str
        self.net = net # str
        self.rltv_x = rltv_x # relative to the component center
        self.rltv_y = rltv_y # relative to the component center
        self.abs_x = abs_x # absolute position on canvas
        self.abs_y = abs_y # absolute position on canvas
    
    def from_dict(self, pin_dict):
        self.pin_name = pin_dict["pin_name"]
        self.anon_name = pin_dict["anon_name"]
        self.component = pin_dict["component"]
        self.net = pin_dict["net"]
        self.rltv_x = pin_dict["rltv_x"]
        self.rltv_y = pin_dict["rltv_y"]
        self.abs_x = pin_dict["abs_x"]
        self.abs_y = pin_dict["abs_y"]
    
    def to_dict(self):
        return {
            "pin_name": self.pin_name,
            "anon_name": self.anon_name,
            "component": self.component, # component refdes
            "net": self.net,
            "rltv_x": self.rltv_x,
            "rltv_y": self.rltv_y,
            "abs_x": self.abs_x,
            "abs_y": self.abs_y
        }

class Net:
    def __init__(self, net_name, anon_name):
        # should be easy to dump and load
        self.net_name = net_name
        self.anon_name = anon_name
        self.pins = [] # should contain [comp_name, pin_name]

    def from_dict(self, net_dict):
        self.net_name = net_dict["net_name"]
        self.anon_name = net_dict["anon_name"]
        self.pins = net_dict["pins"]
    
    def to_dict(self):
        return {
            "net_name": self.net_name,
            "anon_name": self.anon_name,
            "pins": self.pins
        }

class Component:
    def __init__(
        self, 
        name,
        anon_name,
        shape,
        ctr_x,
        ctr_y,
        layer,
        page,
        height,
    ):
        # should be easy to dump and
        self.name = name # REFDES
        self.anon_name = anon_name # anonymous name
        self.shape = shape # should be a polygon, represented as a list of [x, y], x, y are abs positions
        # note that we don't record the rotation of the component, rotation should be 
        # reflected in the shape and pin positions
        self.ctr_x = ctr_x # center x position
        self.ctr_y = ctr_y # center y position
        self.pins = {} # pin_name -> Pin # DYNAMIC DATA STRUCTURE
        self.layer = layer
        self.page = page # schematic page number
        self.fixed = False
        self.height = height # 3d height of the component, i.e. z axis length

    def get_x_length(self):
        # to avoid confusion we don't use "width"
        min_x, max_x = float("inf"), float("-inf")
        for x, y in self.shape:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
        return max_x - min_x
    
    def get_y_length(self):
        # to avoid confusion we don't use "height" or "length"
        min_y, max_y = float("inf"), float("-inf")
        for x, y in self.shape:
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        return max_y - min_y

    def translate_to(self, new_ctrx, new_ctry):
        diff_x = new_ctrx - self.ctr_x
        diff_y = new_ctry - self.ctr_y
        # update the shape
        new_shape = []
        for point in self.shape:
            new_shape.append([point[0] + diff_x, point[1] + diff_y])
        self.shape = new_shape
        # update pins
        for pin_name, pin in self.pins.items():
            pin.abs_x = new_ctrx + pin.rltv_x
            pin.abs_y = new_ctry + pin.rltv_y
        self.ctr_x = new_ctrx
        self.ctr_y = new_ctry

    def rotate(self, radians):
        def rotate_point(x, y, cx, cy, radians):
            x -= cx
            y -= cy
            new_x = x * math.cos(radians) - y * math.sin(radians)
            new_y = x * math.sin(radians) + y * math.cos(radians)
            return new_x + cx, new_y + cy
        # update the shape
        new_shape = []
        for point in self.shape:
            new_shape.append(rotate_point(point[0], point[1], self.ctr_x, self.ctr_y, radians))
        self.shape = new_shape
        # update pins
        for pin_name, pin in self.pins.items():
            pin.abs_x, pin.abs_y = rotate_point(pin.abs_x, pin.abs_y, self.ctr_x, self.ctr_y, radians)
            pin.rltv_x = pin.abs_x - self.ctr_x
            pin.rltv_y = pin.abs_y - self.ctr_y




    def from_dict(self, comp_dict):
        self.name = comp_dict["name"]
        self.anon_name = comp_dict["anon_name"]
        self.shape = comp_dict["shape"]
        self.ctr_x = comp_dict["ctr_x"]
        self.ctr_y = comp_dict["ctr_y"]
        self.layer = comp_dict["layer"]
        self.page = comp_dict["page"]
        self.height = comp_dict["height"]
        self.pins = {}
        for pin_name in comp_dict["pins"]:
            pin = Pin(None, None, None, None, None, None, None, None)
            pin.from_dict(comp_dict["pins"][pin_name])
            self.pins[pin_name] = pin

    def to_dict(self):
        pin_dict = {}
        for pin_name in self.pins:
            pin_dict[pin_name] = self.pins[pin_name].to_dict()
        return {
            "name": self.name,
            "anon_name": self.anon_name,
            "shape": self.shape,
            "ctr_x": self.ctr_x,
            "ctr_y": self.ctr_y,
            "layer": self.layer,
            "page": self.page,
            "height": self.height,
            "pins": pin_dict
        }

    
class PCB:
    def __init__(self):
        self.name = None
        self.components = {} # DYNAMIC DATA STRUCTURE
        self.nets = {} # DYNAMIC DATA STRUCTURE
        self.layers = []
        self.x_range = None # min and max x values, canvas size
        self.y_range = None # min and max y values, canvas size
        
    def import_from_idf(
        self, 
        project_name,
        revision,
        worklib
    ):
        from idf_design import IDFDesign
        idf = IDFDesign(
            project_name=project_name,
            revision=revision,
            worklib=worklib
        )
        self.name = idf.design_name
        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")
        for idx, comp in enumerate(idf.components):
            new_comp = Component(
                name=comp.refdes,
                anon_name=f"C{idx}",
                shape=list(comp.shape.exterior.coords),
                ctr_x=comp.position[0],
                ctr_y=comp.position[1],
                layer=comp.layer,
                page=int(comp.page_number),
                height=comp.height
            )
            min_x = min(min_x, comp.shape.exterior.bounds[0])
            max_x = max(max_x, comp.shape.exterior.bounds[2])
            min_y = min(min_y, comp.shape.exterior.bounds[1])
            max_y = max(max_y, comp.shape.exterior.bounds[3])
            self.components[comp.refdes] = new_comp
            if comp.layer not in self.layers:
                self.layers.append(comp.layer)

        # set the canvas size
        self.x_range = [min_x, max_x]
        self.y_range = [min_y, max_y]

        for idx, netname in enumerate(idf.pstxnet_net_dict):
            clean_netname = netname.replace("'", "")
            new_net = Net(net_name=clean_netname, anon_name=f"N{idx}")
            for ref_pin in idf.pstxnet_net_dict[netname]:
                refdes, pin_name = ref_pin.split("-")
                # ignore irrelevant pins
                if refdes not in self.components: continue
                # look up pin info
                for pin_dict in idf.pm_thickness_dict[refdes]:
                    if pin_name != pin_dict["pin_number"]: continue
                    rltv_x = pin_dict["pin_x"] - pin_dict["center_x"]
                    rltv_y = pin_dict["pin_y"] - pin_dict["center_y"]
                    abs_x = pin_dict["pin_x"]
                    abs_y = pin_dict["pin_y"]
                    new_pin = Pin(
                        pin_name=pin_name,
                        component=refdes,
                        anon_name=f"P{len(self.components[refdes].pins)}",
                        net=clean_netname,
                        rltv_x=rltv_x,
                        rltv_y=rltv_y,
                        abs_x=abs_x,
                        abs_y=abs_y
                    )

                    # add pin info to the net
                    new_net.pins.append([refdes, pin_name])

                    # add pin to the component
                    self.components[refdes].pins[pin_name] = new_pin

            # add net to the pcb design
            self.nets[clean_netname] = new_net

    def to_dict(self):
        comp_dict = {}
        for comp_name in self.components:
            comp_dict[comp_name] = self.components[comp_name].to_dict()
        net_dict = {}
        for net_name in self.nets:
            net_dict[net_name] = self.nets[net_name].to_dict()
        return {
            "name": self.name,
            "components": comp_dict,
            "nets": net_dict,
            "layers": self.layers,
            "x_range": self.x_range,
            "y_range": self.y_range
        }

    def from_dict(self, pcb_dict):
        self.name = pcb_dict["name"]
        self.layers = pcb_dict["layers"]
        self.x_range = pcb_dict["x_range"]
        self.y_range = pcb_dict["y_range"]
        self.components = {}
        for comp_name in pcb_dict["components"]:
            comp = Component(None, None, None, None, None, None, None, None)
            comp.from_dict(pcb_dict["components"][comp_name])
            self.components[comp_name] = comp
        self.nets = {}
        for net_name in pcb_dict["nets"]:
            net = Net(None, None)
            net.from_dict(pcb_dict["nets"][net_name])
            self.nets[net_name] = net


    def visualize(self, figure_path, draw_nets=False, dpi=300):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon

        fig, ax = plt.subplots()

        # plot components
        for comp_name in self.components:
            comp = self.components[comp_name]
            shape = comp.shape
            if comp.layer == "TOP":
                ax.add_patch(MplPolygon(shape, closed=True, edgecolor='#809c13', facecolor='#b5e550', alpha=0.5))
            else:
                ax.add_patch(MplPolygon(shape, closed=True, edgecolor='#005073', facecolor='#71c73c', alpha=0.5))

        # plot pins as black filled circles
        for comp_name in self.components:
            comp = self.components[comp_name]
            for pin_name in comp.pins:
                pin = comp.pins[pin_name]
                ax.plot(pin.abs_x, pin.abs_y, 'ko', markersize=0.1)

        # plot net as grey lines
        if draw_nets:
            for net_name in self.nets:
                net = self.nets[net_name]
                # if the net has les than 2 pins, skip
                if len(net.pins) < 2: continue
                # from the first pin to the second to the last pin
                pin1 = net.pins[0]
                for i in range(1, len(net.pins)):
                    pin2 = net.pins[i]
                    ax.plot(
                        [self.components[pin1[0]].pins[pin1[1]].abs_x, self.components[pin2[0]].pins[pin2[1]].abs_x],
                        [self.components[pin1[0]].pins[pin1[1]].abs_y, self.components[pin2[0]].pins[pin2[1]].abs_y],
                        'k-', linewidth=0.1, alpha=0.3
                    )

        # set the canvas size
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)

        fig.savefig(figure_path, dpi=dpi)
        print("saved figure: ", figure_path)

    
    def visualize_pages(self, figure_dir, dpi=300):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        # if figuure dir doesn't exist, create it
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        
        page_to_comp = {}
        for comp_name, comp in self.components.items():
            if comp.page not in page_to_comp:
                page_to_comp[comp.page] = []
            page_to_comp[comp.page].append(comp_name)
    
        for page in page_to_comp:
            fig, ax = plt.subplots()
            for comp_name in page_to_comp[page]:
                comp = self.components[comp_name]
                shape = comp.shape
                if comp.layer == "TOP":
                    ax.add_patch(MplPolygon(shape, closed=True, edgecolor='#809c13', facecolor='#b5e550', alpha=0.5))
                else:
                    ax.add_patch(MplPolygon(shape, closed=True, edgecolor='#005073', facecolor='#71c73c', alpha=0.5))
                for pin_name in comp.pins:
                    pin = comp.pins[pin_name]
                    ax.plot(pin.abs_x, pin.abs_y, 'ko', markersize=0.1)
            ax.set_xlim(self.x_range)
            ax.set_ylim(self.y_range)
            fig.savefig(os.path.join(figure_dir, f"page_{page}.jpg"), dpi=dpi)
            print("saved page: ", os.path.join(figure_dir, f"page_{page}.jpg"))


    def select_pages(self, page_range):
        new_comps = {}
        new_nets = {}
        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")
        for comp_name in self.components:
            comp = self.components[comp_name]
            if comp.page in page_range:
                new_comps[comp_name] = comp
                min_x = min(min_x, comp.ctr_x - comp.get_x_length() / 2)
                max_x = max(max_x, comp.ctr_x + comp.get_x_length() / 2)
                min_y = min(min_y, comp.ctr_y - comp.get_y_length() / 2)
                max_y = max(max_y, comp.ctr_y + comp.get_y_length() / 2)

        # update the canvas size
        self.x_range = [min_x, max_x]
        self.y_range = [min_y, max_y]

        for net_name in self.nets:
            net = self.nets[net_name]
            new_pins = []
            for pin in net.pins:
                if self.components[pin[0]].page in page_range: # pin[0] is the comp name (REFDES)
                    new_pins.append(pin)
            if new_pins:
                new_net = Net(net_name=net_name, anon_name=net.anon_name)
                new_net.pins = new_pins
                new_nets[net_name] = new_net
        
        self.components = new_comps
        self.nets = new_nets

    def select_layer(self, layer):
        new_comps = {}
        new_nets = {}
        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")
        for comp_name in self.components:
            comp = self.components[comp_name]
            if comp.layer == layer:
                new_comps[comp_name] = comp
                min_x = min(min_x, comp.ctr_x)
                max_x = max(max_x, comp.ctr_x)
                min_y = min(min_y, comp.ctr_y)
                max_y = max(max_y, comp.ctr_y)

        # update the canvas size
        self.x_range = [min_x, max_x]
        self.y_range = [min_y, max_y]
        # stretch the canvas by 1.2x
        self.x_range = [
            self.x_range[0] - 0.1 * (self.x_range[1] - self.x_range[0]),
            self.x_range[1] + 0.1 * (self.x_range[1] - self.x_range[0])
        ]
        self.y_range = [
            self.y_range[0] - 0.1 * (self.y_range[1] - self.y_range[0]),
            self.y_range[1] + 0.1 * (self.y_range[1] - self.y_range[0])
        ]

        for net_name in self.nets:
            net = self.nets[net_name]
            new_pins = []
            for pin in net.pins:
                if self.components[pin[0]].layer == layer:
                    new_pins.append(pin)
            if new_pins:
                new_net = Net(net_name=net_name, anon_name=net.anon_name)
                new_net.pins = new_pins
                new_nets[net_name] = new_net
        
        self.components = new_comps
        self.nets = new_nets