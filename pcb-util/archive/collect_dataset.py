import os

count_comps_file = "./count_comps.txt"

with open(count_comps_file, "r") as f:
    lines = f.readlines()

worklibs = []

for line in lines:
    count, project_number, worklib = line.split(":")
    worklib = worklib.strip()
    # e.g. //syseng/Projects/Bison/Boards/PT285/A00/design/worklib/sub_osfp_power_16x/packaged/pstxprt.dat
    # we want to get //syseng/Projects/Bison/Boards/PT285/A00
    worklib = worklib.split("/design")[0]
    worklibs.append(worklib)

lib_base_path = "//niansongz-arg02-avr/Projects/"

# An example P4 view: 
#         //syseng/Projects/... //niansongz-arg02-avr/Projects/...
out_file = "p4_views.txt"
for worklib in worklibs:
    project_path = worklib.replace("//syseng/Projects/", lib_base_path)
    view_line = f"         {worklib}/... {project_path}/..."
    with open(out_file, "a") as f:
        f.write(view_line + "\n")
