Automated Execution of Energy-Aware Production Schedules Based on Real Industrial Process and Order Data
====================================================================================================================================

The rising share of variable renewable electricity in Germany has increased price volatility, emphasizing
the need for demand response. This study introduces an execution service within Energy-Aware
Production Scheduling (EAPS), using OPC UA for automated schedule implementation. Based on real
industrial data, machine states and energy use are modeled, enabling a cyber-physical system to optimize
makespan, energy costs, and peak load. Simulations show a 6% energy cost reduction compared to
the Shortest Processing Time dispatching rule. Successful integration with four simulated OPC UA
servers confirms system stability, demonstrating the enhanced EAPS architecture's potential for
automated, energy-flexible, and sustainable manufacturing.

**Keywords:** Industrial communication, Job-Shop Scheduling Problem, Demand Response Measures


**See also: Published version on** `Github <https://github.com/PTW-TUDa/Simplified-Implementation-of-Energy-Aware-Production-Scheduling-in-Job-Shops/>`_ **of the original EAPS architecture.**


Installation
------------------------

**This package is tested with Julia 1.10.4 and Python 3.11.9**

To install and use this package you need to have `Julia <https://julialang.org/downloads/>`_ installed. Once this is
done, the package can be activated using the Julia Package manager. To do this, open ``julia`` in in a console, then
type ``]`` to open the package manager. The console should show the ``pkg>`` prompt. Now instantiate the package using:

.. code-block::

    activate .
    instantiate

This should install all required dependencies with the correct versions. Sometimes this process is prone to failure --
in that case you have to install all dependencies manually (see Project.toml for this).

Should you have a different julia version installed, you should first try to resolve the dependencies by running
``Pkg.resolve()``. Should this fail, you can try to update the dependencies by running ``Pkg.update()``, which could
however lead to unexpected behavior, therefore it is recommended to use the correct julia version:

.. code-block::

    activate .
    resolve  #or update
    instantiate

Since this package interacts with the python library eta_utility, you should afterwards install that and ensure that
Julia's PyCall is linked to the correct interpreter. To do this, return to the normal terminal prompt and execute the
following commands.

Now install all project dependencies and let poetry run the install-julia script:

.. code-block::

    poetry install
    poetry run install-julia

The ``install-julia`` command is provided by eta_utility to ensure that PyCall is built correctly. After the
commands complete, you can use this package package by calling one of the starting scripts.

Make sure that the environment variable ``JULIA_PROJECT`` is set either in any sript you execute prior to importing pyjulia or globally.

If you encounter any issues during this process, try removing your caches and starting over

.. code-block::

    rm -rf ~/.julia/compiled/<julia_version_id>

Usage
-----------

After completing the installation process, you can start either of the activation scripts. The ``scheduling.py`` files
in the ``trials`` folders start the production scheduling optimization and the ``TrainModels.jl`` file in the
``modeling`` folder starts the energy model parameter estimation.

Citing this project
--------------------

Please cite this project using the publication:

.. code-block::

    Publication in Review.
