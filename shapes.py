import ast
import matplotlib.pyplot as plt
from dimensions import G19_GEN5

def load_edges(path: str):
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                p1, p2 = ast.literal_eval(line)  # safely parses tuples/lists/numbers
                edges.append((p1, p2))
            except Exception as e:
                raise ValueError(f"Failed to parse line {i}: {line!r}\nError: {e}") from e
    return edges

def generate_enclosure(length, height, slant_length):
    return [
        # Bottom edge
        ((0,0), (length, 0)),
        # Right edge with mag slant
        ((length, 0), (length + slant_length, height)),
        # Top edge
        ((length + slant_length, height), (0, height))
    ]

def generate_trigger_bar(offset, length, height):
    return [
        # Left edge
        ((offset, 0), (offset, height)),
        # Top edge
        ((offset, height), (offset + length, height)),
        # Right edge
        ((offset + length, height), (offset + length, 0))
    ]

def generate_protrusion(start_x, start_y, length, height, slant_length):
    return [
        # Top edge
        ((start_x, start_y + height), (start_x + length + slant_length, start_y + height)),
        # Right edge with slant
        ((start_x + length + slant_length, start_y + height), (start_x + length, start_y)),
        # Bottom edge
        ((start_x + length, start_y), (start_x, start_y))
    ]

def edge_intersection(edgeA, edgeB):
    (x1, y1), (x2, y2) = edgeA
    (x3, y3), (x4, y4) = edgeB

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None  # Parallel lines

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    if 0 <= ua <= 1 and 0 <= ub <= 1:
        return (x1 + ua * (x2 - x1), y1 + ua * (y2 - y1))
    return None

def shape_intersection(shapeA, shapeB):
    for edgeA in shapeA:
        for edgeB in shapeB:
            intersection = edge_intersection(edgeA, edgeB)
            if intersection:
                return intersection
    return None

def draw_shapes(shapes):
    fig, ax = plt.subplots()

    for shape in shapes:
        for (x1, y1), (x2, y2) in shape:
            ax.plot([x1, x2], [y1, y2], linewidth=1)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Shapes")
    ax.grid(True, linewidth=0.5)

    plt.show()

if __name__ == "__main__":
    shapes = [
        generate_enclosure(
            G19_GEN5['enclosure_length'],
            G19_GEN5['enclosure_height'],
            G19_GEN5['enclosure_slant_length']
        ),
        generate_trigger_bar(
            G19_GEN5['trigger_bar_offset'],
            G19_GEN5['trigger_bar_length'],
            G19_GEN5['trigger_bar_height']
        ),
        generate_protrusion(
            G19_GEN5['protrusion_start_x'],
            G19_GEN5['protrusion_start_y'],
            G19_GEN5['protrusion_length'],
            G19_GEN5['protrusion_height'],
            G19_GEN5['protrusion_slant_length']
        )
    ]

    draw_shapes(shapes)
