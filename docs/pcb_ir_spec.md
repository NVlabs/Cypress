# PCB Intermediate Representation (IR) Specification

The Cypress framework utilizes a YAML-based Intermediate Representation (IR) to address the fragmentation of PCB data formats across different EDA tools. This IR provides a portable, extensible, and easy-to-visualize format for capturing essential PCB design information, including components, pins, and nets.

## IR Schema Overview

The IR is structured hierarchically, representing a single PCB design.

### 1. PCB Object
The top-level object representing the entire board.

| Field | Type | Description |
| :--- | :--- | :--- |
| `name` | String | The name of the PCB design. |
| `x_range` | List [min_x, max_x] | The horizontal extent of the board canvas. |
| `y_range` | List [min_y, max_y] | The vertical extent of the board canvas. |
| `layers` | List [String] | List of board layers (e.g., "TOP", "BOTTOM"). |
| `components` | Map {RefDes: Component} | A dictionary mapping component Reference Designators to Component objects. |
| `nets` | Map {NetName: Net} | A dictionary mapping Net names to Net objects. |

### 2. Component Object
Represents an individual component on the PCB.

| Field | Type | Description |
| :--- | :--- | :--- |
| `name` | String | The Reference Designator (e.g., "U1", "R10"). |
| `anon_name` | String | An anonymous identifier for the component. |
| `shape` | List [ [x, y], ... ] | A list of points defining the component's boundary polygon (absolute coordinates). |
| `ctr_x` | Float | The absolute X-coordinate of the component's center. |
| `ctr_y` | Float | The absolute Y-coordinate of the component's center. |
| `layer` | String | The layer the component is placed on ("TOP" or "BOTTOM"). |
| `page` | Integer | The schematic page number where the component originates. |
| `height` | Float | The 3D height of the component (Z-axis). |
| `fixed` | Boolean | Whether the component's position is fixed. |
| `pins` | Map {PinName: Pin} | A dictionary mapping pin names to Pin objects. |

### 3. Pin Object
Represents a connection point on a component.

| Field | Type | Description |
| :--- | :--- | :--- |
| `pin_name` | String | The name of the pin (e.g., "1", "VCC"). |
| `anon_name` | String | An anonymous identifier for the pin. |
| `component` | String | The Reference Designator of the parent component. |
| `net` | String | The name of the net this pin is connected to. |
| `rltv_x` | Float | X-offset relative to the component center. |
| `rltv_y` | Float | Y-offset relative to the component center. |
| `abs_x` | Float | Absolute X-coordinate on the board. |
| `abs_y` | Float | Absolute Y-coordinate on the board. |

### 4. Net Object
Represents an electrical connection between multiple pins.

| Field | Type | Description |
| :--- | :--- | :--- |
| `net_name` | String | The name of the net. |
| `anon_name` | String | An anonymous identifier for the net. |
| `pins` | List [ [RefDes, PinName], ... ] | A list of pin references connected by this net. |

---

## Translation Tools

The `pcb-util/` directory contains several tools for converting between the IR and common EDA formats:

*   **Bookshelf**: `pcb-util/bookshelf_converter.py` - Supports bidirectional conversion between IR and the Bookshelf format used by many placement tools.
*   **KiCad**: `pcb-util/kicad_converter.py` - Supports bidirectional conversion between IR and `.kicad_pcb` files.
*   **IDF (Intermediate Data Format)**: `pcb-util/idf_design.py` - Supports importing designs from IDF (common in mechanical CAD/ECAD exchange).
*   **NS-Place**: `pcb-util/nsplace_converter.py` - Specifically handles conversion from the KiCad 4 format used by the NS-Place tool.

## CPA (Component Placement Automation) Flow

Cypress performs component placement by leveraging the IR as a bridge between high-level EDA formats and the GPU-accelerated placement engine. The flow for IDF-based designs is as follows:

1.  **Import**: The design is imported from IDF and Allegro data files into the Cypress IR using `IDFDesign` and `PCB.import_from_idf()`.
2.  **Bookshelf Export**: Since the core placement engine (DREAMPlace) consumes Bookshelf files, the IR is exported to `.nodes`, `.nets`, `.pl`, and `.scl` files using the `BookshelfExporter`.
3.  **Placement Optimization**: The `CypressEngine` loads these Bookshelf files and executes the `PlacementEngine`. This step uses GPU-accelerated non-linear optimization to minimize wirelength and congestion.
4.  **Result Extraction**: Final coordinates are retrieved using `engine.get_final_positions()`, which can then be used to update the original design or for further analysis.


## Visualization

The `PCB` class in `pcb-util/pcb.py` includes built-in visualization capabilities:
*   `visualize(figure_path, draw_nets=True)`: Generates a 2D plot of the component placement and optionally the flight lines for nets using Matplotlib.
*   `visualize_pages(figure_dir)`: Generates separate plots for each schematic page, helping to understand the logical grouping of components.

## Sample IR File

A sample YAML IR file can be found at: [sample_pcb.yaml](sample_pcb.yaml)
