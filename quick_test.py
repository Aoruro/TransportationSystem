"""Quick test to verify the project works"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("="*60)
    print("TSP Project - Quick Test")
    print("="*60)

    # 1. Test imports
    print("\n[1/4] Testing imports...")
    try:
        import numpy as np
        print("[OK] numpy imported")
        
        from data.data_processor import DataProcessor, TSPInstance
        print("[OK] data module imported")
        
        from search.astar import AStarSolver
        from search.bfs import BFSSolver
        from search.ucs import UCSSolver
        print("[OK] search module imported")
        
        from ml.learning_astar import LearningAStar
        print("[OK] ml module imported")
        
        from csp.tsptw_solver import TSPTWSolver
        print("[OK] csp module imported")
        
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        sys.exit(1)

    # 2. Test data loading
    print("\n[2/4] Testing data loading...")
    try:
        processor = DataProcessor(data_path="tsp_instances_dataset.csv")
        instances = processor.load_from_csv()
        print(f"[OK] Loaded {len(instances)} instances")
        
        filtered = processor.preprocess(max_cities=20)
        print(f"[OK] Filtered to {len(filtered)} instances (N <= 20)")
        
        if not filtered:
            print("[FAIL] No valid instances found!")
            sys.exit(1)
        
        inst = filtered[0]
        print(f"[OK] Using instance: {inst.name} (N={inst.n})")
        
    except Exception as e:
        print(f"[FAIL] Data loading failed: {e}")
        sys.exit(1)

    # 3. Test A* algorithm
    print("\n[3/4] Testing A* algorithm...")
    try:
        solver = AStarSolver(inst.dist_matrix)
        result = solver.solve()
        
        print(f"[OK] Success: {result['success']}")
        print(f"[OK] Cost: {result['cost']:.2f}")
        print(f"[OK] Nodes expanded: {result['nodes_expanded']}")
        print(f"[OK] Time: {result['time']:.4f}s")
        
    except Exception as e:
        print(f"[FAIL] A* failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Test solve_with_callback
    print("\n[4/4] Testing solve_with_callback...")
    try:
        callback_count = 0
        def test_callback(node_info):
            nonlocal callback_count
            callback_count += 1
        
        solver2 = AStarSolver(inst.dist_matrix)
        result2 = solver2.solve_with_callback(callback=test_callback)
        
        print(f"[OK] Callback called {callback_count} times")
        print(f"[OK] Result success: {result2['success']}")
        print(f"[OK] Search tree: {len(result2['search_tree'])} nodes")
        
    except Exception as e:
        print(f"[FAIL] solve_with_callback failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*60)
    print("[OK] All quick tests passed.")
    print("="*60)

if __name__ == "__main__":
    main()
