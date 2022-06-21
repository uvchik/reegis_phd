#! /usr/bin/env python

from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


g = "@https://github.com/"
setup(
    name="my_reegis",
    version="0.0.1",
    author="Uwe Krien",
    author_email="uwe.krien@rl-institut.de",
    description="A reegis heat and power model of Berlin.",
    long_description=read("README.rst"),
    long_description_content_type="text/x-rst",
    packages=find_packages(),
    package_dir={"my_reegis": "my_reegis"},
    install_requires=[
        "oemof.solph == 0.4.1",
        "pandas == 1.1.4",
        "requests == 2.25",
        "matplotlib == 3.3.3",
        "graphviz == 0.15",
        "descartes == 1.1.0",
        "demandlib{0}oemof/demandlib/archive/v0.1.7b1.zip".format(g),
        "pytz == 2020.4",
        "numpy == 1.22.0",
        "reegis{0}reegis/reegis/archive/phd.zip".format(g),
        "deflex{0}reegis/deflex/archive/phd.zip".format(g),
        "berlin_hp{0}reegis/berlin_hp/archive/phd.zip".format(g),
        "geopandas == 0.8.1",
        "scenario_builder{0}reegis/scenario_builder/archive/phd.zip".format(g),
        "oemof-visio{0}oemof/oemof-visio/archive/v0.0.1b1.zip".format(g),
    ],
    extras_require={"dummy": ["oemof"]},
    entry_points={
        "console_scripts": [
            "phd_figures = reegis_phd.figures.figures:main",
            "reproduce_phd_optimisation = reegis_phd.reproduce:main",
        ]
    },
    package_data={
        "reegis_phd": [
            os.path.join("data", "static", "*.csv"),
            os.path.join("data", "static", "*.txt"),
            os.path.join("data", "geometries", "*.csv"),
            os.path.join("data", "geometries", "*.geojson"),
            os.path.join("data", "figures", "*.svg"),
            os.path.join("data", "figures", "*.png"),
            os.path.join("data", "figures", "*.graphml"),
            "*.ini",
        ]
    },
)
