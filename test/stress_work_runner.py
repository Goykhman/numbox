""" Stress-testing `Work` graphs (on Apple M1 Pro 16 Gb)

**********************
***** Single Run *****
**********************

*** 1,000 input nodes, ~ 2,000 total nodes ***

Stats when making image:

First run over a clear cache:

Execution of do_create_nodes took 22.207s
Execution of do_make_image took 88.870s
Execution of do_calculate took 18.521s

Second run loading cache:

Execution of do_create_nodes took 0.159s
Execution of do_make_image took 0.593s
Execution of do_calculate took 0.148s

Stats without making image:

First run over a clear cache:

Execution of do_create_nodes took 23.089s
Execution of do_calculate took 18.254s

~0.6 Gb memory used, cache size ~380 Mb,
25 Mb for `create_nodes` the rest is `Work` nodes calculations

Second run loading cache:

Execution of do_create_nodes took 0.169s
Execution of do_calculate took 0.148s

~0.1 Gb memory used

*** 5,000 input nodes, ~ 10,000 total nodes ***

Stats when making image:

First run over a clear cache:

Execution of do_create_nodes took 817.861s
Execution of do_make_image took 631.145s
Execution of do_calculate took 161.130s

Second run loading cache:

Execution of do_create_nodes took 0.953s
Execution of do_make_image took 2.671s
Execution of do_calculate took 0.874s

Stats without making image:

First run over a clear cache:

Execution of do_create_nodes took 820.860s
Execution of do_calculate took 156.426s

~5.75 Gb memory used, cache size ~3.05 Gb,
25 Mb for `create_nodes` the rest is `Work` nodes calculations

Second run loading cache:

Execution of do_create_nodes took 0.973s
Execution of do_calculate took 0.875s

~1.5 Gb memory used


**********************
**** Multiple Run ****
**********************

1,000 input nodes, ~2,000 total nodes
10,000 entities

First run over a clear cache:

Execution of do_create_nodes took 23.113s
Execution of run took 69.796s
~1.7 Gb memory used, ~560 Mb cache disk space

Second run loading cache:

Execution of do_create_nodes took 0.159s
Execution of run took 3.779s (0.664s for 1,000 entities)
~0.7Gb memory used (about the same for 1,000 entities)
"""
from numbox.utils.timer import timer
from test.stress_work import multiple_run, single_run


@timer
def do_make_image(w_):
    from numbox.core.work.print_tree import make_image
    w_image = make_image(w_)
    print(w_image)


if __name__ == "__main__":
    num_of_inputs = 1000
    num_of_entities = 10000
    total_data = multiple_run(num_of_inputs, num_of_entities)
    print(f"total_data = {total_data}")
    w = single_run(num_of_inputs)
    print(f"w.data = {w.data}")
    # do_make_image(w)
