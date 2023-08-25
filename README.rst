###########
PyObjDetect
###########


..
   .. image:: https://img.shields.io/pypi/v/pyobjdetect.svg
           :target: https://pypi.python.org/pypi/pyobjdetect

.. image:: https://readthedocs.org/projects/pyobjdetect/badge/?version=latest
        :target: https://pyobjdetect.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Playing with pytorch object detection tutorial


* Free software: GNU General Public License v3
* Documentation: https://pyobjdetect.readthedocs.io.


********
Features
********

* Pytorch's Object Detection tutorial
* Pytorch's Transfer learning tutorial

*************
Prerequisites
*************
* `Pytorch Object Detection Tutorial`_
* `Dataset [object detection]`_
* `Pytorch Transfer Learning Tutorial`_
* `Dataset [transfer learning]`_
* `Cuda Toolkit 11.7`_

.. _Pytorch Object Detection Tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
.. _Dataset [object detection]: https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
.. _Pytorch Transfer Learning Tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
.. _Dataset [transferlearning]: https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
.. _`Cuda Toolkit 11.7`: https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network

*****
Setup
*****

=================
Environment setup
=================
First, run the `source scripts/env.sh` to get setup. It will ask to setup a python virtual environment if one does not already exist

.. code-block:: bash

    # to use the default virtual environment (.env)
    source ./scripts/env.sh

    # instead, if you want to use your custom virtual environment
    source ./scripts/env.sh <path_to_custom venv>

Then, install the package in editable mode

.. code-block:: bash

    ./scripts/build.sh install

=============
Training Data
=============

Download training data.

.. code-block:: bash

    # download the Penn-Fudan dataset
    wget https://www.cis.upenn.edu/\~jshi/ped_html/PennFudanPed.zip

    # extract it in the current folder
    unzip PennFudanPed.zip -d `echo ${ODT_DATA_DIR}/` && rm PennFudanPed.zip

That's it!

======================
Running a demo example
======================

.. code-block:: bash

    python pyobjdetect/model/demomodel.py


**********
Developing
**********
When you return to the project, run the `source` command to set up the environment and reactivate your virtual environment

.. code-block:: bash

    # to use the default virtual environment (.env)
    source ./scripts/env.sh

    # if you want to use your custom virtual environment
    source ./scripts/env.sh <path_to_custom venv>


***************
Building wheels
***************
Run the following from the root directory of this repo. It will generate a wheel file in `repo_root_dir/dist`.

.. code-block:: bash

    python -m build .


*******
Credits
*******

This package was created with Cookiecutter_ and the `draconus21/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`draconus21/cookiecutter-pypackage`: https://gitlab.com/draconus21/cookiecutter-pypackage
