
import zipfile
import fnmatch
import os

projects = "/home/niansongz/scratch/syseng/Projects"

# automatically find all worklibs by walking projects, find this pattern:
# */design/worklib/*
worklibs = []
for root, dirs, files in os.walk(projects):
    for dir in dirs:
        if dir == "nv_pcb_export":
            nv_pcb_export = os.path.join(root, dir)
            parent = os.path.dirname(nv_pcb_export)
            worklibs.append(parent)

def find_file_recursive(directory, pattern):
    """
    Recursively searches for the first file that matches the pattern in the given directory.

    Args:
    - directory (str): The directory to start the search from.
    - pattern (str): The pattern of the file to search for (e.g., '*.txt').

    Returns:
    - str: The full path to the file if found, None otherwise.
    """
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                return os.path.join(root, filename)
    return None


# worklibs = [
#     "/home/niansongz/scratch/pcb-util/inputs/PB201/A00/design/worklib/pb201_a00",
#     "/home/niansongz/scratch/pcb-util/inputs/PT008/A00/design/worklib/pt008_a00",
#     "/home/niansongz/scratch/pcb-util/inputs/PT014/A00/design/worklib/pt014_a00",
#     "/home/niansongz/scratch/pcb-util/inputs/XDR/Crocodile/PT060/A00/design/worklib/pt060_a00",
#     "/home/niansongz/scratch/pcb-util/inputs/XDR/Crocodile/PT061/A00/design/worklib/pt061_a00",
#     "/home/niansongz/scratch/pcb-util/inputs/XDR/Crocodile/PT069/A00/design/worklib/pt069_a00",
#     "/home/niansongz/scratch/pcb-util/inputs/XDR/Crocodile/PT125/A00/design/worklib/pt125_a00",
#     "/home/niansongz/scratch/pcb-util/inputs/XDR/Crocodile/PT142/A00/design/worklib/pt142_a00"
# ]

def find_file_in_zip(zip_path, pattern):
    """
    Searches for a file matching the pattern within a zip archive.

    Args:
    - zip_path (str): The path to the zip file.
    - pattern (str): The pattern of the file to search for (e.g., '*.txt').

    Returns:
    - str: The name of the file if found, None otherwise.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if fnmatch.fnmatch(file_name, pattern):
                    return file_name
    except zipfile.BadZipFile:
        print(f"Error: The file '{zip_path}' is not a valid zip file.")
        return None

    # print(f"Error: No file matching pattern '{pattern}' found in zip file '{zip_path}'")
    return None

def report_completeness(worklib):
    patterns = ["pstxnet.dat", "pstxprt.dat", "page.map", "module_order.dat"]
    idf_patterns = ["*.emn", "*.emp"]
    print("\nChecking worklib: ", worklib)
    for pattern in patterns:
        files = find_file_recursive(worklib, pattern)
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
            # print(f"Thickness file '{thickness_file}' found in '{zip_612_file}'")
            pass
        else:
            # print(f"Thickness file not found in '{zip_612_file}'")
            return False

    else:
        # no thickness file found
        return False

    return True
    # print("Done.")

complete = []
incomplete = []

for worklib in worklibs:
    if report_completeness(worklib):
        complete.append(worklib)
    else:
        incomplete.append(worklib)

with open("complete_worklibs.txt", "w") as f:
    for worklib in complete:
        f.write(worklib + "\n")

# complete = report_completeness("/home/niansongz/scratch/syseng/Projects/DGX/Boards/P4479_TH500_SXM7_GB1x2/A00/design/worklib/p4479_a00")
# print("Complete: ", complete)