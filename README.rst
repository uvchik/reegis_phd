This package contains all scripts to reproduce the phd-work of Uwe Krien:

`The pdf-Document of the thesis (german).
<https://github.com/uvchik/reegis_phd/blob/master/monographie_krien_Bewertungskriterien_Regionalmodell.pdf>`_

.. image:: https://zenodo.org/badge/137771506.svg
   :target: https://zenodo.org/badge/latestdoi/137771506

.. contents::
    :depth: 1
    :local:
    :backlinks: top

Initiated packages used in `reegis_phd`
=======================================

The phd-Work includes the following projects, initiated by the author.
The status is meant by November 2020.

`oemof.solph <https://github.com/oemof/oemof-solph>`_
+++++++++++++++++++++++++++++++++++++++++++++++++++++

A model generator for energy system modelling and optimisation (LP/MILP).

* Status: active
* contribution: oemof-developer-group
* Used Version: 0.4.1

`demandlib <https://github.com/oemof/demandlib>`_
+++++++++++++++++++++++++++++++++++++++++++++++++

Creating heat and power demand profiles from annual values.

* Status: active
* contribution: oemof-developer-group
* Used Version: v0.1.7b1

`windpowerlib <https://github.com/wind-python/windpowerlib>`_
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
The windpowerlib is a library that provides a set of functions and classes to
calculate the power output of wind turbines.

* Status: active
* contribution: oemof-developer-group
* Used Version: v0.2.1b1

`oemof-visio <https://github.com/oemof/oemof-visio>`_
+++++++++++++++++++++++++++++++++++++++++++++++++++++

Visualisation package of oemof.

* Status: active
* contribution: oemof-developer-group
* Used Version: v0.0.1b1

`reegis <https://github.com/reegis/reegis>`_
++++++++++++++++++++++++++++++++++++++++++++

Open geospatial data model for energy systems.

* Status: active
* contribution: `@uvchik <https://github.com/uvchik>`_ (mainly)
* Used Version: phd

`deflex <https://github.com/reegis/deflex>`_
++++++++++++++++++++++++++++++++++++++++++++

Simple heat and power model of Germany.

* Status: active
* contribution: `@uvchik <https://github.com/uvchik>`_ (mainly)
* Used Version: phd

`scenario_builder <https://github.com/reegis/scenario_builder>`_
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Helper functions to create historical and future scenarios for energy system modelling.

* Status: experimental/active
* contribution: `@uvchik <https://github.com/uvchik>`_ (mainly)
* Used Version: phd

`berlin_hp <https://github.com/reegis/berlin_hp>`_
++++++++++++++++++++++++++++++++++++++++++++++++++

Simple heat and power model of Berlin.

* Status: not active
* contribution: `@uvchik <https://github.com/uvchik>`_
* Used Version: phd

`Open_eQuarterPy <https://github.com/reegis/Open_eQuarterPy>`_
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Stand alone python script of the Open_eQuarter qgis plugin. It can be used to estimate the annual heat demand of buidlings using open data.

* Status: not active
* contribution: `@uvchik <https://github.com/uvchik>`_
* Used Version: phd


Installation
============

If you have a working Python 3 environment, use pypi to install the latest the
package.

.. code-block::

    pip install https://github.com/uvchik/reegis_phd/archive/phd.zip


Some required packages need the installation of additional binary packages on
your operating system.

On a Linux Debian system you can use the following command to solve the
requirements beforehand.

.. code-block::

    sudo apt-get install python3-dev proj-bin libproj-dev libgeos-dev python3-tk libspatialindex-dev virtualenv

For other operating systems follow the installation instructions of each
package.


Basic usage
===========

Basically two scripts are provided: `phd_figures` and
`reproduce_phd_optimisation`. With `phd_figures` it is possible to plot all
figures that are based on results. For all other figures the source code of the
image (svg or graphml) is downloaded.

For all 6.x figures the results of the scenarios are needed. To create these
results the `reproduce_phd_optimisation` script can be used.

phd_figures
+++++++++++

.. NOTE::

    The plot of some figures may take some minutes up to hours on the
    **first run**, because the needed data is calculated in the background. The
    data is stored on your hard drive to speed up the following plots.

    The most time-consuming step is the calculation of feed-in time series.

Pass the number of the figure to plot and store it. If no path is given the
default path is ``$HOME/reegis``.

To plot e.g. figure 3.5 use the following command:

.. code-block::

    phd_figures 3.5

To define a directory for the stored figures a path can be passed:

.. code-block::

    phd_figures 3.5 /home/username/my_figures

It is also possible to create all figures of the work and store it to a given
directory. This may take some time especially on the first run (see above).

.. code-block::

    phd_figures all /home/username/my_figures


reproduce_phd_optimisation
++++++++++++++++++++++++++

To solve large scenarios a RAM of up to 24 GB is necessary. The script uses
parallelisation and you have pass the fraction (0 to 1) of the cores to be
used for the optimisation. Be aware that the scenarios need up to 24 GB of
RAM so that two large parallel scenarios may need 48 GB and so on. To use one
core on a PC just pass a small number:

.. code-block::

    reproduce_phd_optimisation 0.01

The default path is ``$HOME/reegis``, to use a different path type:

.. code-block::

    reproduce_phd_optimisation 0.01 /your/path/for/the/results

License
============

Copyright (c) 2020 Uwe Krien

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.