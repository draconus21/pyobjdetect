#!/usr/bin/env python

"""Tests for `pyobjdetect` package."""


import unittest
from click.testing import CliRunner

from pyobjdetect import pyobjdetect
from pyobjdetect import cli


class TestPyobjdetect(unittest.TestCase):
    """Tests for `pyobjdetect` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'pyobjdetect.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
