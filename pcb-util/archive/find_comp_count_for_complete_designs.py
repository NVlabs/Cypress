

complete_designs = "/home/niansongz/scratch/pcb-util/info/complete_worklibs.txt"
# e.g. /home/niansongz/scratch/syseng/Projects/E3643/A00/design/worklib/e3643_a00



count_comps = "/home/niansongz/scratch/pcb-util/info/count_comps.txt"
# e.g. 184 : E4734_B00 : //syseng/Projects/E4734/B00/design/worklib/e4734_b00/packaged/pstxprt.dat


with open(complete_designs, 'r') as f:
    complete_designs_lines = f.readlines()

# for each line in complete_designs_lines, extract the "design/worklib/e3643_a00" part
# use regex
import re

name_to_path = {}

for line in complete_designs_lines:
    match = re.search(r"design/worklib/\w+_\w+", line)
    if match:
        name_to_path[match.group()] = line


# read all of them
with open(count_comps, 'r') as f:
    count_comps_lines = f.readlines()

# search each pattern in count_comps, if match, return the line
# use regex
lines = []
for name in name_to_path:
    for line in count_comps_lines:
        match = re.search(name, line)
        if match:
            count = line.split(':')[0].strip()
            line = count + " : " + name_to_path[name]
            lines.append(line)
            break

# write to a new file
with open("../info/complete_designs_with_comp_count.txt", 'w') as f:
    for line in lines:
        f.write(line)
