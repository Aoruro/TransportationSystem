"""
visualization module initialization file

Interactive UI and visualization components:
- TSPVisualizer: Interactive GUI visualizer with Tkinter
- SimpleVisualizer: Static matplotlib visualization for non-GUI environments

Reference notes:
- Tkinter and Matplotlib are reused for UI and plotting infrastructure.
- Full bibliographic details and reuse rationale are in REFERENCES.md.
"""
from .ui import TSPVisualizer, SimpleVisualizer

__all__ = ['TSPVisualizer', 'SimpleVisualizer']
