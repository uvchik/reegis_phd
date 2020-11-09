import os

from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from reegis.tools import download_file

from reegis_phd import config as cfg

NAMES = {
    "lignite": "Braunkohle",
    "natural_gas": "Gas",
    "oil": "Öl",
    "hard_coal": "Steinkohle",
    "netto_import": "Stromimport",
    "other": "sonstige",
    # 'nuclear': 'Atomkraft',
}


def create_subplot(default_size, **kwargs):
    size = kwargs.get("size", default_size)
    return plt.figure(figsize=size).add_subplot(1, 1, 1)


def show_download_image(name, file_types):
    create_subplot((12, 4.4))
    fn = os.path.join(
        cfg.get("paths", "figure_source"), "{0}.png".format(name)
    )
    img = mpimg.imread(fn)
    plt.imshow(img)
    plt.axis("off")
    plt.subplots_adjust(left=0, top=0.93, bottom=0, right=1)

    url = (
        "https://raw.githubusercontent.com/uvchik/reegis_phd/master/"
        "reegis_phd/data/figures/{0}.{1}"
    )
    fn = os.path.join(cfg.get("paths", "figures"), "{0}.{1}")
    for suffix in file_types:
        download_file(fn.format(name, suffix), url.format(name, suffix))
    plt.title(
        "Image source downloaded to: {0}".format(fn.format(name, file_types[0]))
    )
    return "{0}.png".format(name), None
