import os
import sys
import shutil
import csv
import zipfile
import bs_utils
from bs_utils import find_file_recursive, find_file_in_zip

# we import it here because it depends on NVIDIA tools
# we only need this when we import from IDF
nv_sda_path = "/home/niansongz/scratch/syseng/syseng/eda/app/prd/sda"
# test if nv_sda_path exists
if not os.path.exists(nv_sda_path):
    raise ValueError(f"NVIDIA SDA path is not set, {nv_sda_path} does not exist.")
else:
    print(f"NVIDIA SDA path is set to {nv_sda_path}.")

sys.path.append(os.path.join(nv_sda_path, "nv_da/shared_libraries"))
sys.path.append(os.path.join(nv_sda_path, "nv_da/sda_webserver"))
sys.path.append(os.path.join(nv_sda_path, "nv_ecad_sch/nv_nvConn/backend"))

from nv_parsers.idf_parser import IDF
from nvConn import parse_net


class IDFDesign:
    def __init__(
        self,
        project_name,
        revision,
        worklib,
    ):
        self.project_name = project_name
        self.revision = revision
        self.design_name = f"{project_name}_{revision}"
        self.worklib = worklib

        self.prepare_files()
        self.parse_idf()
        
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
            file = find_file_recursive(self.worklib, name)
            if file:
                shutil.copy(file, interm_folder)
            else:
                raise ValueError(f"File {name} not found in {self.worklib}")
        self.pstxnet = os.path.join(interm_folder, "pstxnet.dat")
        self.pstxprt = os.path.join(interm_folder, "pstxprt.dat")
        self.page_map = os.path.join(interm_folder, "page.map")
        self.module_order = os.path.join(interm_folder, "module_order.dat")
                # find IDF files
        idf_patterns = [f"*{self.revision.upper()}.emn", f"*{self.revision.upper()}.emp"]
        for idf in idf_patterns:
            if find_file_recursive(interm_folder, idf):
                continue # skip if already exists
            file = find_file_recursive(
                os.path.join(self.worklib, "nv_pcb_export"), idf
            )
            if file:
                shutil.copy(file, interm_folder)
            else:
                raise ValueError(
                    f"IDF file {idf} not found in {self.worklib}"
                )

        self.emn_file = find_file_recursive(interm_folder, "*.emn")
        self.emp_file = find_file_recursive(interm_folder, "*.emp")

        # find thickness file
        # already exists?
        previously_copied = find_file_recursive(interm_folder, "*pm_thickness.csv")
        if previously_copied:
            self.pm_thickness_csv_file = previously_copied
            return

        zip_612_file = find_file_recursive(self.worklib, "612*.zip")
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


    def parse_idf(self):
        self.page_per_ref_dict, self.refdeses_per_page_dict = bs_utils.main(
            pstxprt=self.pstxprt,
            page_map=self.page_map,
            module_order=self.module_order,
            project_number="",
            revision="",
            destination_folder=self.worklib,
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
            if comp.refdes in self.page_per_ref_dict:
                page_number = self.page_per_ref_dict[comp.refdes]["pdf_page_number"][0]
            else:
                page_number = 0
            comp.page_number = page_number
            self.component_dict[comp.refdes] = comp

        self.pm_thickness_dict = self.parse_pm_thickness_csv_file(
            self.pm_thickness_csv_file
        )
    
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
                center_x = float(row[2])
                center_y = float(row[3])
                pin_number = row[4]
                pin_x = float(row[5])
                pin_y = float(row[6])
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