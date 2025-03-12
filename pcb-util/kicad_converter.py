import os

import kiutils
from kiutils.board import Board
from kiutils.footprint import Footprint, Pad
from kiutils.items.common import Position, PageSettings
from kiutils.items.common import Net as kiNet
from kiutils.items.fpitems import FpPoly

from pcb import PCB, Net, Component, Pin


class KiCadConverter:
    def __init__(self, target_path, anonynous=False):
        self.target_path = target_path
        self.anonynous = anonynous
        self.board = None
        self.pcb = None
        self.net_dict = {} # net name to kicad net obj

    def to_kicad(self, pcb):
        # get the folder name of output filename (output_path)
        output_folder = os.path.dirname(self.target_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # produce kicad 7 files
        self.pcb = pcb
        self.board = Board().create_new()
        
        # page settings
        page_width = int(pcb.x_range[1] - pcb.x_range[0])
        page_height = int(pcb.y_range[1] - pcb.y_range[0])
        self.board.paper = PageSettings(
            paperSize="User",
            width=page_width * 20,
            height=page_height * 20,
        )


        # KiCAD's net connection is defined with pin
        # i.e. we declare what nets we have, then define how pins connect to the nets when we define pins
        for idx, net_name in enumerate(pcb.nets):
            if "GND" in net_name or "GROUND" in net_name:
                continue
            net = pcb.nets[net_name]
            kicad_net = kiNet(idx, net.anon_name if self.anonynous else net_name)
            self.board.nets.append(kicad_net)
            self.net_dict[net_name] = kicad_net
        
        for cname, comp in pcb.components.items():
            comp_name = comp.anon_name if self.anonynous else cname
            # create footprint
            fp = Footprint().create_new(
                library_id=comp_name,
                value=comp_name,
                reference=comp_name,
            )
            # set layer
            fp.layer = "F.Cu" if comp.layer == "TOP" else "B.Cu"
            # set position
            fp.position = Position(comp.ctr_x - pcb.x_range[0], comp.ctr_y - pcb.y_range[0])
            # add polygon shape
            coords = []
            for x, y in comp.shape:
                coords.append(Position(x-comp.ctr_x, y-comp.ctr_y)) # relative position
            fp_layer = "F.SilkS" if comp.layer == "TOP" else "B.SilkS"
            fp_poly = FpPoly(coordinates=coords, fill="solid", layer=fp_layer)
            fp.graphicItems.append(fp_poly)                        

            for pin_name, pin in comp.pins.items():
                if pin.net not in self.net_dict:
                    continue
                # create pad
                pad = Pad(
                    number=pin_name,
                    position=Position(pin.rltv_x, pin.rltv_y),
                    layers=["F.Cu" if comp.layer == "TOP" else "B.Cu"],
                    size=Position(0.1, 0.1),
                )
                pad.net = self.net_dict[pin.net]
                fp.pads.append(pad)
            self.board.footprints.append(fp)
        
        self.board.to_file(self.target_path)
            

    def from_kicad(self):
        # only update the placement of the PCB
        self.board = Board().from_file(self.target_path)
        filename = os.path.basename(self.target_path)
        name = filename.split(".")[0]
        self.pcb = PCB()
        for kicad_net in self.board.nets:
            net_name = kicad_net.name
            if net_name == '': continue
            self.pcb.nets[net_name] = Net(net_name, None)

        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        for idx, fp in enumerate(self.board.footprints):
            comp_name = f"C{idx}"
            shape = []
            for gi in fp.graphicItems:
                if not isinstance(gi, FpPoly): continue
                for point in gi.coordinates:
                    shape.append([point.X + fp.position.X, point.Y + fp.position.Y])
            layer = "TOP" if fp.layer == "F.Cu" else "BOTTOM"
            ctr_x = fp.position.X
            ctr_y = fp.position.Y
            min_x = min(min_x, ctr_x)
            min_y = min(min_y, ctr_y)
            max_x = max(max_x, ctr_x)
            max_y = max(max_y, ctr_y)
            self.pcb.components[comp_name] = Component(
                comp_name, None, shape, ctr_x, ctr_y, layer, None, None
            )
            for pad in fp.pads:
                pin_name = pad.number
                rltv_x = pad.position.X
                rltv_y = pad.position.Y
                abs_x = ctr_x + rltv_x
                abx_y = ctr_y + rltv_y
                if pad.net is None:
                    # net_name = "GND"
                    continue
                else:
                    net_name = pad.net.name
                # add pin to component
                self.pcb.components[comp_name].pins[pin_name] = Pin(
                    pin_name, None, comp_name, net_name, rltv_x, rltv_y, abs_x, abx_y
                )
                # add pin to net
                self.pcb.nets[net_name].pins.append([comp_name, pin_name])
        # sanity check
        # for net_name in self.pcb.nets:
            # net = self.pcb.nets[net_name] 
            # assert len(net.pins) > 0, f"Net {net_name} has no pins"

        # expand the canvas by 1.2x
        min_x -= 0.1 * (max_x - min_x)
        max_x += 0.1 * (max_x - min_x)
        min_y -= 0.1 * (max_y - min_y)
        max_y += 0.1 * (max_y - min_y)
        self.pcb.x_range = [min_x, max_x]
        self.pcb.y_range = [min_y, max_y]

        return self.pcb
