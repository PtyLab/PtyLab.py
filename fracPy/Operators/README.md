# Physics module

This module contains all the things that involve physics.

For instance, `propagator_singlemode.py` contains all the methods you need to propagate a field from A to B. If you're an end user, you will most likely not need to change anything here.

If you're a developer, this is where the core stuff happens. All the methods in BaseReconstructor that involve any kind of physics should just call a physics.* function with the right arguments.

