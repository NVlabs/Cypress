import os
import fnmatch


pcb_benchmark_root = "/pcb-benchmark"


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

f = open("run_tuner.sh", "w")

for i in range(1, 11): # range from 1 to 10
    bench_name = f"small-{i}"
    # find the .aux file in the benchmark directory
    aux_file = find_file_recursive(f"{pcb_benchmark_root}/{bench_name}/bookshelf", "*.aux")
    assert aux_file is not None, f"Could not find .aux file for benchmark {bench_name}"

    cmd = f"""
./tuner/run_tuner.sh 1 1 \
	test/tune/pcb-configspace.json \
	{aux_file} \
	test/pcb/tuner-ppa.json \"\" \
	20 2 0 0 10 \
	./tuner \
	./results/tuner/{bench_name}
"""
    
    f.write(cmd)
    f.write("\n")

f.close()
