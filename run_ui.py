"""Run TSP Visualization UI"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualization.ui import TSPVisualizer

if __name__ == "__main__":
    print("Starting TSP Visualizer...")
    print("Features:")
    print("  - Load custom TSP instances")
    print("  - Select algorithm: BFS/UCS/A*/Learning A*")
    print("  - Real-time search tree visualization")
    print("  - Path animation")
    print("  - Adjustable lambda parameter")
    print()
    
    app = TSPVisualizer()
    app.run()
