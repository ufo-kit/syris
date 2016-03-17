"""Example utility functions."""

import argparse
import matplotlib.pyplot as plt


def get_default_parser(description):
    """Default argument parser with *description*"""
    return argparse.ArgumentParser(description=description,
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def show(image, title=''):
    """Show *image* with *title* on its own figure."""
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.colorbar()
    plt.show(False)
