#! /usr/bin/env python

from setuptools import setup


github = "@https://github.com/"
setup(
    name="my_reegis",
    version="0.0.1",
    author="Uwe Krien",
    author_email="uwe.krien@rl-institut.de",
    description="A reegis heat and power model of Berlin.",
    package_dir={"my_reegis": "my_reegis"},
    install_requires=[
        "oemof.solph == 0.4.1",
        "pandas == 1.1.4",
        "requests == 2.25",
        "matplotlib == 3.3.3",
        "demandlib{0}oemof/demandlib/archive/v0.1.7b1.zip".format(github),
        "pytz == 2020.4",
        "numpy == 1.19.4",
        "reegis{0}reegis/reegis/archive/phd.zip".format(github),
        "deflex{0}reegis/deflex/archive/phd.zip".format(github),
        "berlin_hp{0}reegis/berlin_hp/archive/phd.zip".format(github),
        "geopandas == 0.8.1",
    ],
    extras_require={"dummy": ["oemof"]},
    entry_points={
        "console_scripts": [
            "phd_figures = reegis_phd.figures.figures:main",
            "reproduce_phd_optimisation = reegis_phd.reproduce:main",
        ]
    },
)
