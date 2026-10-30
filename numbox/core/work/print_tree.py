from io import StringIO

VERTICAL_SEP = "|"
HORIZONTAL_BIT = "-"
HORIZONTAL_SEP = HORIZONTAL_BIT * 2


class ImmutableItemDict(dict):
    """A dictionary in which each key can be set only once; setting a key again raises ``ValueError``.

    The layout uses it to detect two items placed in one cell of the grid.
    """
    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(f"Already stored {value} in {key}")
        super().__setitem__(key, value)


def calculate_col_widths(graph):
    """Return the width of each column of the grid ``graph``: the length of the longest item in the column.

    ``graph`` maps (row, column) pairs to strings.
    """
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
    """Return the tree of ``node`` laid out on a grid, with the numbers of rows and columns and the column widths.

    The grid maps (row, column) pairs to strings. Each item is padded to its column's width, with ``-`` when the next
    cell of its row holds an item and with spaces otherwise; a ``|`` is always padded with spaces. The inputs of a node
    are placed two columns to its right, one below another, and a node that the tree reaches along several paths
    appears once on each path.
    """
    graph = ImmutableItemDict()
    _build_graph(node, graph, 0, 0)
    col_widths = calculate_col_widths(graph)
    # The padding below sets each cell a second time, and ImmutableItemDict raises ValueError for a key that is already
    # set. The padding therefore writes to a plain copy of the grid.
    graph = dict(graph)
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
    """Return a text drawing of the tree of ``node`` and of the nodes it depends on.

    ``node`` is a :class:`~numbox.core.work.node.Node` or a :class:`~numbox.core.work.work.Work` node. Deeper nodes are
    drawn further to the right, and the inputs of a node are drawn one below another. A node that the tree reaches
    along several paths appears once on each path. The text begins with a newline, and no row ends in a space.
    """
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
