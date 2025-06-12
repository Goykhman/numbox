from io import StringIO

VERTICAL_SEP = "|"
HORIZONTAL_BIT = "-"
HORIZONTAL_SEP = HORIZONTAL_BIT * 2


class ImmutableItemDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(f"Already stored {value} in {key}")
        super().__setitem__(key, value)


def calculate_col_widths(graph):
    col_widths = {}
    for (row, col), item in graph.items():
        col_widths[col] = max(col_widths.setdefault(col, 0), len(item))
    return col_widths


def _build_graph(node, graph, row, col):
    graph[(row, col)] = str(node)
    node_inputs = node.get_inputs_names()
    if len(node_inputs) == 0:
        return 1
    graph[(row, col + 1)] = HORIZONTAL_SEP
    subtree_height = 0
    for i, input_ in enumerate(node_inputs):
        if i > 0:
            graph[(row + subtree_height, col + 2)] = VERTICAL_SEP
            r = row + subtree_height - 1
            while r > 0 and not graph.get((r, col + 2), None):
                graph[(r, col + 2)] = VERTICAL_SEP
                r -= 1
            subtree_height += 1
        subtree_height += _build_graph(node.get_input(i), graph, row + subtree_height, col + 2)
    return subtree_height


def make_graph(node):
    graph = ImmutableItemDict()
    _build_graph(node, graph, 0, 0)
    col_widths = calculate_col_widths(graph)
    graph = dict(graph)  # make it mutable
    num_rows = 0
    num_columns = 0
    for (row, col), item in graph.items():
        if (row, col + 1) in graph and item != VERTICAL_SEP:
            graph[(row, col)] = item.ljust(col_widths[col], HORIZONTAL_BIT)
        else:
            graph[(row, col)] = item.ljust(col_widths[col], " ")
        num_rows = max(num_rows, row + 1)
        num_columns = max(num_columns, col + 1)
    return graph, num_rows, num_columns, col_widths


def make_image(node):
    graph, num_rows, num_columns, col_widths = make_graph(node)
    image = StringIO()
    image.write("\n")
    for row in range(num_rows):
        row_image = StringIO()
        for col in range(num_columns):
            item = graph.get((row, col), None)
            if item is None:
                row_image.write(" " * col_widths[col])
            else:
                row_image.write(item)
        image.write(f"{row_image.getvalue().rstrip()}\n")
    image = image.getvalue().rstrip()
    return image
