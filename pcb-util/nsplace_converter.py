import sexpdata
from pprint import pprint
from pcb import PCB, Net, Component, Pin

board_start_str = """
(kicad_pcb (version 20171130) (host pcbnew "(5.1.2-1)-1")

  (general
    (thickness 1.6)
  )

  (page A4)
  (layers
    (0 Top signal)
    (31 Bottom signal)
    (32 B.Adhes user)
    (33 F.Adhes user)
    (34 B.Paste user)
    (35 F.Paste user)
    (36 B.SilkS user)
    (37 F.SilkS user)
    (38 B.Mask user)
    (39 F.Mask user)
    (40 Dwgs.User user)
    (41 Cmts.User user)
    (42 Eco1.User user)
    (43 Eco2.User user)
    (44 Edge.Cuts user)
    (45 Margin user)
    (46 B.CrtYd user)
    (47 F.CrtYd user)
    (48 B.Fab user)
    (49 F.Fab user)
  )
  (setup
    (last_trace_width 0.25)
    (trace_clearance 0.2)
    (zone_clearance 0.508)
    (zone_45_only no)
    (trace_min 0.2)
    (via_size 0.8)
    (via_drill 0.4)
    (via_min_size 0.4)
    (via_min_drill 0.3)
    (uvia_size 0.3)
    (uvia_drill 0.1)
    (uvias_allowed no)
    (uvia_min_size 0.2)
    (uvia_min_drill 0.1)
    (edge_width 0.05)
    (segment_width 0.2)
    (pcb_text_width 0.3)
    (pcb_text_size 1.5 1.5)
    (mod_edge_width 0.12)
    (mod_text_size 1 1)
    (mod_text_width 0.15)
    (pad_size 1.524 1.524)
    (pad_drill 0.762)
    (pad_to_mask_clearance 0.051)
    (solder_mask_min_width 0.25)
    (aux_axis_origin 0 0)
    (visible_elements FFFFEF7F)
    (pcbplotparams
      (layerselection 0x010fc_ffffffff)
      (usegerberextensions false)
      (usegerberattributes false)
      (usegerberadvancedattributes false)
      (creategerberjobfile false)
      (excludeedgelayer true)
      (linewidth 0.100000)
      (plotframeref false)
      (viasonmask false)
      (mode 1)
      (useauxorigin false)
      (hpglpennumber 1)
      (hpglpenspeed 20)
      (hpglpendiameter 15.000000)
      (psnegative false)
      (psa4output false)
      (plotreference true)
      (plotvalue true)
      (plotinvisibletext false)
      (padsonsilk false)
      (subtractmaskfromsilk false)
      (outputformat 1)
      (mirror false)
      (drillshape 1)
      (scaleselection 1)
      (outputdirectory ""))
  )
"""


class KiCad4Converter:

    def __init__(self, target_path, anonymous=False):
        self.target_path = target_path
        self.anonymous = anonymous
        self.pcb = None
        self.board = None
        self.net_dict = {} # net name to kicad net obj

    def to_kicad(self, pcb):
        # get the folder name of output filename (output_path)
        output_folder = os.path.dirname(self.target_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.pcb = pcb
        f = open(self.target_path, "w")
        f.write(board_start_str)

        # print nets
        netid = 0
        for netname in pcb.nets.keys():
            f.write(f"  (net {netid} {netname})\n")
            netid += 1
        
        # print net classes
        f.write("  (net_class Default \"\"\n")
        for netname in pcb.nets.keys():
            f.write(f"    (add_net {netname})\n")
        f.write("  )\n")

        # print footprint
                


        f.close()

    def from_kicad(self):
        # start with empty pcb
        self.pcb = PCB()
        with open(self.target_path, "r") as f:
            kicad_content = f.read()
        kicad_content = sexpdata.loads(kicad_content)

        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")

        for element in kicad_content:
            if element[0] == sexpdata.Symbol("net"):
                net_name = element[-1]
                net_name = net_name.value() if isinstance(net_name, sexpdata.String) else net_name
                if net_name == "": continue
                assert net_name is not None
                self.pcb.nets[net_name] = Net(net_name, net_name)

            if element[0] == sexpdata.Symbol("module"):
                # this element is a component
                comp_name = None
                shape = []
                layer = None
                ctr_x = None
                ctr_y = None
                pin_dict = {}
                for item in element:
                    if item[0] == sexpdata.Symbol("layer"):
                        layer = "TOP" if item[1] == sexpdata.Symbol("Top") else "BOTTOM"
                    if item[0] == sexpdata.Symbol("at"):
                        ctr_x = item[1]
                        ctr_y = item[2]
                        assert isinstance(ctr_x, float)
                        assert isinstance(ctr_y, float)
                    if item[0] == sexpdata.Symbol("fp_text") and item[1] == sexpdata.Symbol("reference"):
                        comp_name = item[2].value()
                    if item[0] == sexpdata.Symbol("fp_poly"):
                        for points in item[1]:
                            if points[0] == sexpdata.Symbol("xy"):
                                x = points[1]
                                y = points[2]
                                assert isinstance(x, float)
                                assert isinstance(y, float)
                                abs_x = x + ctr_x
                                abs_y = y + ctr_y
                                min_x = min(min_x, abs_x)
                                min_y = min(min_y, abs_y)
                                max_x = max(max_x, abs_x)
                                max_y = max(max_y, abs_y)
                                shape.append([abs_x, abs_y])
                    if item[0] == sexpdata.Symbol("pad"):
                        pin_name = item[1]
                        pin_name = pin_name.value() if isinstance(pin_name, sexpdata.String) else pin_name
                        rltv_x = item[4][1]
                        rltv_y = item[4][2]
                        assert isinstance(rltv_x, float)
                        assert isinstance(rltv_y, float)
                        abs_x = ctr_x + rltv_x
                        abs_y = ctr_y + rltv_y
                        
                        net_name = item[-1][-1].value()
                        pin_dict[pin_name] = Pin(pin_name, pin_name, comp_name, net_name, rltv_x, rltv_y, abs_x, abs_y)
                        # self.pcb.components[comp_name].pins[pin_name] = Pin(pin_name, None, comp_name, net_name, rltv_x, rltv_y, abs_x, abs_y)
        
                # add component to pcb
                self.pcb.components[comp_name] = Component(comp_name, comp_name, shape, ctr_x, ctr_y, layer, 0, None)
                for pin_name, pin in pin_dict.items():
                    self.pcb.components[comp_name].pins[pin_name] = pin
                    self.pcb.nets[pin.net].pins.append([comp_name, pin_name])


        self.pcb.x_range = [min_x, max_x]
        self.pcb.y_range = [min_y, max_y]
        return self.pcb
        


# kicad_path = "/home/nz264/shared/ns-place/output.small-7_kicad4.kicad_pcb"
# converter = KiCad4Converter(kicad_path)
# pcb = converter.from_kicad()
# pcb.visualize("test_kicad4.png", draw_nets=True)


