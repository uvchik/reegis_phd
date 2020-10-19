from matplotlib import pyplot as plt


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
