===========
PyObjDetect
===========


..
   .. image:: https://img.shields.io/pypi/v/pyobjdetect.svg
           :target: https://pypi.python.org/pypi/pyobjdetect

.. image:: https://readthedocs.org/projects/pyobjdetect/badge/?version=latest
        :target: https://pyobjdetect.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Playing with pytorch object detection tutorial


* Free software: GNU General Public License v3
* Documentation: https://pyobjdetect.readthedocs.io.


Features
--------

* TODO

* [ Reference ](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
* [ Dataset ](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip)

Prerequisites
-------------
* [Cuda Toolkit 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network)

Environment setup
-----------------
First, run the `source scripts/env.sh` to get setup. It will ask to setup a python virtual environment if one does not already exist

.. code-block:: bash

    # to use the default virtual environment (.env)
    source ./scripts/env.sh

    # instead, if you want to use your custom virtual environment
    source ./scripts/env.sh <path_to_custom venv>

Then, install the package in editable mode

.. code-block:: bash

    ./scripts/build.sh install

That's it!

Developping
-----------
When you return to the project, run the `source` command to set up the environment and reactivate your virtual environment

.. code-block:: bash

    # to use the default virtual environment (.env)
    source ./scripts/env.sh

    # if you want to use your custom virtual environment
    source ./scripts/env.sh <path_to_custom venv>


Credits
-------

This package was created with Cookiecutter_ and the `draconus21/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`draconus21/cookiecutter-pypackage`: https://gitlab.com/draconus21/cookiecutter-pypackage
