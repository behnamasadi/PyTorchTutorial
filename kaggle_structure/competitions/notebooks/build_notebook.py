"""Assemble a polished, educational Kaggle .ipynb from (markdown, code) sections.

Produces a self-contained notebook (markdown explanations + runnable code cells)
ready to publish for community upvotes. Used to build each medical notebook.
"""
import json
import sys


def build(path, cells):
    nb_cells = []
    for kind, src in cells:
        if kind == "md":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
        else:
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                             "outputs": [], "source": src})
    nb = {"cells": nb_cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(path, "w"), indent=1)
    print("wrote", path, "-", len(cells), "cells")


if __name__ == "__main__":
    print("import and call build(path, cells)")
    sys.exit(0)
