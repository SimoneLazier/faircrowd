import matplotlib.pyplot as plt

bw = {
    "COLORS": ["#222222", "#444444", "#666666", "#888888", "#aaaaaa", "#cccccc"],
    "MARKERS": ["o", "s", "v", "^", "D", "P"],
    "LINES": ["-", "--", "-.", ":", "-", "--"],
    "SMALL_SIZE": 22,
    "MEDIUM_SIZE": 24,
    "BIGGER_SIZE": 26,
    "FIGSIZE": (8, 6),
}

colors = {
    "COLORS": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    "MARKERS": ["o", "s", "v", "^", "D", "P"],
    "LINES": ["-", "--", "-.", ":", "-", "--"],
    "SMALL_SIZE": 22,
    "MEDIUM_SIZE": 24,
    "BIGGER_SIZE": 26,
    "FIGSIZE": (8, 6),
}

poli_colors = {
    "COLORS": ["#5C8A99", "#5C996B", "#99945C", "#995C5C", "#965C99", "#5F5C99"],
    "MARKERS": ["o", "s", "v", "^", "D", "P"],
    "LINES": ["-", "--", "-.", ":", "-", "--"],
    "SMALL_SIZE": 22,
    "MEDIUM_SIZE": 24,
    "BIGGER_SIZE": 26,
    "FIGSIZE": (12, 6),
}

themes = {"bw": bw, "colors": colors, "poli_colors": poli_colors}
theme = bw


def set_theme(theme_name: str = "bw"):
    """
    Sets the plots theme, possible values are 'bw' (black and white) and 'colors'. Default: 'bw'
    """
    theme = themes.get(theme_name)
    if theme is None:
        raise ValueError("Theme not found!")

    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=theme.get("COLORS"),
        marker=theme.get("MARKERS"),
        linestyle=theme.get("LINES"),
    )
    plt.rc("figure", figsize=theme.get("FIGSIZE"))
    plt.rc("font", family="Times New Roman", size=theme.get("SMALL_SIZE"))
    plt.rc("axes", titlesize=theme.get("BIGGER_SIZE"))
    plt.rc("axes", labelsize=theme.get("MEDIUM_SIZE"))
    plt.rc("xtick", labelsize=theme.get("SMALL_SIZE"))
    plt.rc("ytick", labelsize=theme.get("SMALL_SIZE"))
    plt.rc("legend", fontsize=theme.get("SMALL_SIZE"))
    plt.rc("figure", titlesize=theme.get("BIGGER_SIZE"))


def get_theme():
    return theme
