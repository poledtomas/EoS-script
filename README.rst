==============
EoS - script
==============

Python script for EoS project.

Process description
==================


Create Python virtual environment::

   python3 -m venv --clear venv
   ./venv/bin/pip install --upgrade pip
   ./venv/bin/pip install -r requirements.txt

To run scripts, use for example::

./venv/bin/python inventer.py  

Contribution guidelines
=======================

This repository uses Ruff_ to ensure consistent code style and for linting.
Shell scripts use Prettier_ for
formatting. Make sure to run ::

   ruff check --fix .
   ruff format .
   prettier --write --ignore-unknown "**/*"

Alternatively, set up pre-commit_ to take care of these tasks. First, install
pre-commit::

   pip install pre-commit

Then, after cloning the repository, install the Git hook scripts::

   pre-commit install

.. _Ruff: https://github.com/astral-sh/ruff
.. _Prettier: https://github.com/prettier/prettier
.. _pre-commit: https://pre-commit.com
