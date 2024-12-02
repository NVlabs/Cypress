import os

from kiutils.board import Board
from kiutils.footprint import Footprint, Pad
from kiutils.items.common import Position, Net, PageSettings
from kiutils.items.fpitems import FpPoly   


def dequote(s):
    s = s.strip()
    s = s.replace('"', '')
    s = s.replace("'", '')
    return s


def convert_kicad_version(input_path, output_path):

    # get the folder name of output filename (output_path)
    output_folder = os.path.dirname(output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    board = Board().from_file(input_path)

    f = open(output_path, "w")
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

    f.write(board_start_str)

    net_dict = {}
    for idx, net in enumerate(board.nets):
        netname = dequote(net.name)
        if net.name == "":
            netname = '\"\"'
        net_dict[netname] = idx

    # print nets
    for netname, netid in net_dict.items():
        f.write(f"  (net {netid} {netname})\n")

    # print net classes
    f.write("  (net_class Default \"\"\n")
    for netname, _ in net_dict.items():
        f.write(f"    (add_net {netname})\n")
    f.write("  )\n")

    # print footprints
    for idx, footprint in enumerate(board.footprints):
        layer = "Top" if footprint.layer == "F.Cu" else "Bottom"
        polylayer = "F.Fab" if footprint.layer == "F.Cu" else "B.Fab"
        f.write(f"  (module {dequote(footprint.libId)} (layer {layer}) (tedit 0) (tstamp 0)\n")
        x, y = footprint.position.X, footprint.position.Y
        # x, y = 0, 0
        f.write(f"   (at {x} {y})\n")
        
        # print reference and names
        ref_string = f"""
    (fp_text
      reference
      C{idx}
      (at {x} {y})
      (layer F.SilkS)
      hide
      (effects (font (size 1.27 1.27) (thickness 0.15))))
    (fp_text
      value
      C{idx}
      (at {x} {y})
      (layer F.SilkS)
      hide
      (effects (font (size 1.27 1.27) (thickness 0.15))))
"""
        f.write(ref_string + "\n")
        # print poly
        for polygon in footprint.graphicItems:
            if not isinstance(polygon, FpPoly):
                continue
            f.write("    (fp_poly (pts ")
            for coords in polygon.coordinates:
                x, y = coords.X, coords.Y
                # print .3f
                f.write(f"(xy {x:.3f} {y:.3f}) ")
            f.write(f") (layer {polylayer}) (width 0.1))\n")
        # print pads
        for pad in footprint.pads:
            f.write(f"    (pad {dequote(pad.number)} {pad.type} {pad.shape} (at {pad.position.X:.3f} {pad.position.Y:.3f}) (size {pad.size.X:.3f} {pad.size.Y:.3f}) (layers {layer} *.Paste *.Mask)\n")
            pads_net = pad.net
            if pads_net:
                correct_number = net_dict[dequote(pads_net.name)]
                f.write(f"      (net {correct_number} {dequote(pads_net.name)})\n")
            # end pad
            f.write(f"    )\n")
        # end of footprint
        f.write("  )\n\n")
    
    # minx, miny, maxx, maxy
    minx = float('inf')
    miny = float('inf')
    maxx = float('-inf')
    maxy = float('-inf')
    for footprint in board.footprints:
        minx = min(minx, footprint.position.X)
        miny = min(miny, footprint.position.Y)
        maxx = max(maxx, footprint.position.X)
        maxy = max(maxy, footprint.position.Y)
    
    # add 20% margin
    margin = 0.2
    minx -= (maxx - minx) * margin
    miny -= (maxy - miny) * margin
    maxx += (maxx - minx) * margin
    maxy += (maxy - miny) * margin

    f.write(f"  (gr_line (start {minx:.3f} {miny:.3f}) (end {minx:.3f} {maxy:.3f}) (layer Edge.Cuts) (width 0.15) (tstamp 0))\n")
    f.write(f"  (gr_line (start {minx:.3f} {maxy:.3f}) (end {maxx:.3f} {maxy:.3f}) (layer Edge.Cuts) (width 0.15) (tstamp 0))\n")
    f.write(f"  (gr_line (start {maxx:.3f} {maxy:.3f}) (end {maxx:.3f} {miny:.3f}) (layer Edge.Cuts) (width 0.15) (tstamp 0))\n")
    f.write(f"  (gr_line (start {maxx:.3f} {miny:.3f}) (end {minx:.3f} {miny:.3f}) (layer Edge.Cuts) (width 0.15) (tstamp 0))\n")

    # end of board
    f.write(")\n")

    f.close()


if __name__ == "__main__":
    input_path = "./outputs/PB310_kicad/p39.kicad_pcb"
    output_path = "./outputs/PB310_kicad/p39_legacy.kicad_pcb"
    convert_kicad_version(input_path, output_path)