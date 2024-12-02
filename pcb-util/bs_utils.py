import sys
import os
from tempfile import mkdtemp
import fnmatch
import zipfile

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
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for file_name in zip_ref.namelist():
                if fnmatch.fnmatch(file_name, pattern):
                    return file_name
    except zipfile.BadZipFile:
        print(f"Error: The file '{zip_path}' is not a valid zip file.")
        return None

    return None

def get_pages(pages):
    pages_list = []
    pages = pages.replace(" ", "")
    for p in pages.split(","):
        page_range = p.split("-")
        pages_list += (
            [f"{x}" for x in range(int(page_range[0]), int(page_range[-1]) + 1)]
            if len(page_range) > 1
            else page_range
        )
    return sorted(pages_list, key=int)


def write_file(out_file, data):
    with open(out_file, "w") as f:
        for line in data:
            f.write(line + "\n")
    print(f"File {out_file} has been written.")


def read_file(f1):
    # print ("f1  ", f1)
    with open(f1, 'r')as f:
        return [i.strip().split() for i in f.readlines()]


def map_instance_to_page_number(module_order_file):
    instance_page_number_dict = {}
    file_content = read_file(module_order_file)

    for idx,line in enumerate(file_content[1:]):
        if not line:
            continue
        if str(line[0]).startswith("@"):
            instance_page_number_dict[str.lower(line[0])] = line[3]

    return instance_page_number_dict


def map_p_path_to_sequential(page_map_file):
    seq_p_path_dict = {}
    file_content = read_file(page_map_file)

    for idx,line in enumerate(file_content):
        if not line:
            continue
        seq_p_path_dict[int(line[1])] = idx + 1
    return seq_p_path_dict

# TODO: take in account all pages that a refdes appears on
def find_page_number_in_pstxprt(pstxprt_file, instance_page_number_dict, seq_p_path_dict):

    page_per_ref_dict = {}
    refdeses_per_page_dict = {}
    pstxprt_content = read_file(pstxprt_file)

    skipped = False
    for idx,line in enumerate(pstxprt_content[1:]):
        if not line:
            continue
        if line[0] == "PART_NAME":
            ref = pstxprt_content[idx+2][0]
            top_page_number = 'NOT FOUND'
            pdf_page_number = 'NOT FOUND'

            page_per_ref_dict[ref] = {}
            page_per_ref_dict[ref]["top_page_number"] = []
            page_per_ref_dict[ref]["pdf_page_number"] = []

        # go over file pstxprt.dat and extract for each ref , the TOP page number from P_PATH + PHYS_PAGE + instance    
        elif str(line[0]).startswith("'@") and not skipped:
            instance = str(line[0][1:-1]).lower().rsplit(':', 1)[0]

        elif str(line[0]).startswith("P_PATH") and not skipped:
            
            p_path_number = (str(line[0]).lower().split(":")[1].split("_")[0]).replace("page", '')
            try:
                top_page_number = seq_p_path_dict[int(p_path_number)]
            except:
                # This could happen if file page.map not found
                top_page_number = 'NOT FOUND'
            if not top_page_number in page_per_ref_dict[ref]["top_page_number"]:
                page_per_ref_dict[ref]["top_page_number"].append(top_page_number)

        elif str(line[0]).startswith("PHYS_PAGE") and not skipped:
            phys_page_number = str(line[0]).split("'")[1]
            try:
                page_for_instance = instance_page_number_dict[instance]
                pdf_page_number = str(int(phys_page_number) + int(page_for_instance) - 1)
                
            except:
                pdf_page_number = top_page_number
            
            if not pdf_page_number in page_per_ref_dict[ref]["pdf_page_number"]:
                page_per_ref_dict[ref]["pdf_page_number"].append(pdf_page_number)


            if pdf_page_number in refdeses_per_page_dict:
                if not ref in refdeses_per_page_dict[pdf_page_number]:
                    refdeses_per_page_dict[pdf_page_number].append(ref)
            else:
                refdeses_per_page_dict[pdf_page_number] = [ref]
        
    return page_per_ref_dict, refdeses_per_page_dict


def main(pstxprt, page_map, module_order, project_number, revision, destination_folder):
    if not destination_folder:
        destination_folder = mkdtemp()
        print(f"No destination_folder passed. destination_folder set as: {destination_folder}")

    # need 3 files to find page number
    if pstxprt and page_map and module_order:
        pass
    else:
        raise ValueError('Need all 3 file pstxprt and page_map and module_order to find page number')
    
    instance_page_number_dict = map_instance_to_page_number(module_order)
    seq_p_path_dict = map_p_path_to_sequential(page_map)
    
    page_per_ref_dict, refdeses_per_page_dict = find_page_number_in_pstxprt(pstxprt, instance_page_number_dict, seq_p_path_dict)
    return page_per_ref_dict, refdeses_per_page_dict

