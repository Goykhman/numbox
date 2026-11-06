"""Run the stress test of ``stress_work`` on a tree of 1,000 end nodes and print the results.

Run the script from the repository root with ``python -m test.stress_work_runner``. It runs ``multiple_run`` for
10,000 entities and prints their totals, and then it runs ``single_run`` and prints the value of the root. The
``timer`` decorator logs the duration of each timed step. ``do_make_image`` draws the tree with ``make_image``, prints
the drawing and logs its duration. Its call at the end of the script is commented out, and the measurements with the
drawing below were taken with the call enabled.

The measurements below were taken on an Apple M1 Pro with 16 GB of memory. A tree of 1,000 end nodes has about 2,000
nodes in total, and a tree of 5,000 end nodes has about 10,000. A first run started from an empty cache, and a second
run loaded the compiled code from the cache.

Single run, 1,000 end nodes
---------------------------

The first run with the drawing logged::

    Execution of do_create_nodes took 22.207s
    Execution of do_make_image took 88.870s
    Execution of do_calculate took 18.521s

The second run with the drawing logged::

    Execution of do_create_nodes took 0.159s
    Execution of do_make_image took 0.593s
    Execution of do_calculate took 0.148s

The first run without the drawing used about 0.6 GB of memory. Its cache occupied about 380 MB on disk, of which
``create_nodes`` used 25 MB and the compiled calculation of the ``Work`` nodes used the rest. It logged::

    Execution of do_create_nodes took 23.089s
    Execution of do_calculate took 18.254s

The second run without the drawing used about 0.1 GB of memory and logged::

    Execution of do_create_nodes took 0.169s
    Execution of do_calculate took 0.148s

Single run, 5,000 end nodes
---------------------------

The first run with the drawing logged::

    Execution of do_create_nodes took 817.861s
    Execution of do_make_image took 631.145s
    Execution of do_calculate took 161.130s

The second run with the drawing logged::

    Execution of do_create_nodes took 0.953s
    Execution of do_make_image took 2.671s
    Execution of do_calculate took 0.874s

The first run without the drawing used about 5.75 GB of memory. Its cache occupied about 3.05 GB on disk, of which
``create_nodes`` used 25 MB and the compiled calculation of the ``Work`` nodes used the rest. It logged::

    Execution of do_create_nodes took 820.860s
    Execution of do_calculate took 156.426s

The second run without the drawing used about 1.5 GB of memory and logged::

    Execution of do_create_nodes took 0.973s
    Execution of do_calculate took 0.875s

Multiple run, 1,000 end nodes and 10,000 entities
-------------------------------------------------

The first run used about 1.7 GB of memory and about 560 MB of cache on disk, and it logged::

    Execution of do_create_nodes took 23.113s
    Execution of run took 69.796s

The second run used about 0.7 GB of memory, about as much as a run with 1,000 entities used, and it logged::

    Execution of do_create_nodes took 0.159s
    Execution of run took 3.779s

With 1,000 entities, ``run`` took 0.664s in the second run.
"""
import logging
import sys

from numbox.utils.timer import timer
from test.stress_work import multiple_run, single_run


@timer
def do_make_image(w_):
    """Draw the tree of ``w_`` with ``make_image`` and print the drawing."""
    from numbox.core.work.print_tree import make_image
    w_image = make_image(w_)
    print(w_image)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    num_of_inputs = 1000
    num_of_entities = 10000
    print("multiple run", file=sys.stderr)
    total_data = multiple_run(num_of_inputs, num_of_entities)
    print(f"total_data = {total_data}")
    print("single run", file=sys.stderr)
    w = single_run(num_of_inputs)
    print(f"w.data = {w.data}")
    # do_make_image(w)
