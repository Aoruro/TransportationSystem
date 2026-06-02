"""
visualization module initialization file

Interactive UI and visualization components:
- TSPVisualizer: Interactive GUI visualizer with Tkinter
- SimpleVisualizer: Static matplotlib visualization for non-GUI environments

References:
- Tkinter for interactive GUI
- Matplotlib for plotting
"""
from .ui import TSPVisualizer, SimpleVisualizer

__all__ = ['TSPVisualizer', 'SimpleVisualizer']