"""Temporary script to get the ray dashboard url for the current experiment.

TODO figure out a more convenient way.
"""

import ray

context = ray.init(address="auto", ignore_reinit_error=True)

print(context)
