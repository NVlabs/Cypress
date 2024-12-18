import sexpdata
from pprint import pprint
from pcb import PCB, Net, Component, Pin

# kicad_path = "/home/nz264/shared/ns-place/output.small-9_kicad4.kicad_pcb"

# with open(kicad_path, "r") as f:
#     kicad_content = f.read()

# kicad_content = sexpdata.loads(kicad_content)

# access all nets
# for element in kicad_content:
#     if element[0] == sexpdata.Symbol("module"):
#         pprint(element)
#         # break
#         # this element is a component
#         for item in element:
#             if item[0] == sexpdata.Symbol("layer"):
#                 print(f"layer: {item}")
#             if item[0] == sexpdata.Symbol("at"):
#                 print(f"placement: ({item[1]}, {item[2]})")
#             if item[0] == sexpdata.Symbol("fp_text") and item[1] == sexpdata.Symbol("reference"):
#                 print(f"name: {item[2].value()}")
#             if item[0] == sexpdata.Symbol("fp_poly"):
#                 print(f"polygon: {item[1]}")
#             if item[0] == sexpdata.Symbol("pad"):
#                 print(f"pad: {item[1]}")
#                 print(f"pad position: {item[4]}")
#                 print(f"pad net: {item[-1]}")


class KiCad4Converter:

    def __init__(self, target_path, anonymous=False):
        self.target_path = target_path
        self.anonymous = anonymous
        self.pcb = None
        self.board = None
        self.net_dict = {} # net name to kicad net obj

    def to_kicad(self, pcb):
        pass


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
                self.pcb.nets[net_name] = Net(net_name, None)

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
                        pin_dict[pin_name] = Pin(pin_name, None, comp_name, net_name, rltv_x, rltv_y, abs_x, abs_y)
                        # self.pcb.components[comp_name].pins[pin_name] = Pin(pin_name, None, comp_name, net_name, rltv_x, rltv_y, abs_x, abs_y)
        
                # add component to pcb
                self.pcb.components[comp_name] = Component(comp_name, None, shape, ctr_x, ctr_y, layer, None, None)
                for pin_name, pin in pin_dict.items():
                    self.pcb.components[comp_name].pins[pin_name] = pin
                    self.pcb.nets[pin.net].pins.append([comp_name, pin_name])


        self.pcb.x_range = [min_x, max_x]
        self.pcb.y_range = [min_y, max_y]
        return self.pcb
        


kicad_path = "/home/nz264/shared/ns-place/output.small-7_kicad4.kicad_pcb"
converter = KiCad4Converter(kicad_path)
pcb = converter.from_kicad()
pcb.visualize("test_kicad4.png", draw_nets=True)
