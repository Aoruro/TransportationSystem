"""Visualization state regression tests."""

import unittest
from unittest.mock import patch

import numpy as np

from data.data_processor import TSPInstance
from visualization.ui import TSPVisualizer


class _Label:
    def __init__(self):
        self.text = ""

    def config(self, **kwargs):
        self.text = kwargs.get("text", self.text)


class _Root:
    def update_idletasks(self):
        pass


class _Axes:
    def clear(self):
        pass

    def scatter(self, *_args, **_kwargs):
        pass

    def annotate(self, *_args, **_kwargs):
        pass

    def plot(self, *_args, **_kwargs):
        pass

    def set_title(self, *_args, **_kwargs):
        pass


class _Canvas:
    def draw(self):
        pass


class TestVisualizerState(unittest.TestCase):
    """UI state tests that do not require a Tk window."""

    def test_search_animation_batches_more_nodes_at_higher_speed(self):
        """Test that the speed slider reduces redraw frequency at high values."""
        self.assertEqual(TSPVisualizer._search_animation_settings(1), (1, 198))
        self.assertEqual(TSPVisualizer._search_animation_settings(50), (25, 100))
        self.assertEqual(TSPVisualizer._search_animation_settings(100), (100, 1))

    def test_reset_resumes_future_searches(self):
        """Test that reset clears pause instead of toggling it."""
        visualizer = object.__new__(TSPVisualizer)
        visualizer._search_run_id = 0
        visualizer.is_paused = True
        visualizer.search_after_id = None
        visualizer.animation_id = None
        visualizer.current_path = [0, 1, 0]
        visualizer.search_tree = [object()]
        visualizer.displayed_nodes = 1
        visualizer.current_instance = None
        visualizer.status_label = _Label()

        visualizer._reset()

        self.assertFalse(visualizer.is_paused)
        self.assertEqual(visualizer.current_path, None)
        self.assertEqual(visualizer.search_tree, [])

    def test_train_ml_model_from_current_instance(self):
        """Test the standalone UI path for configuring Learning A*."""
        visualizer = object.__new__(TSPVisualizer)
        visualizer.current_instance = TSPInstance(
            "small",
            np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 1.5]], dtype=float)
        )
        visualizer.root = _Root()
        visualizer.status_label = _Label()
        visualizer.ml_model = None
        reset_calls = []
        visualizer._reset = lambda: reset_calls.append(True)

        visualizer._train_ml_model()

        self.assertIsNotNone(visualizer.ml_model)
        self.assertEqual(reset_calls, [True])
        self.assertIn("ML model trained", visualizer.status_label.text)

    def test_switching_instance_discards_trained_model(self):
        """Test that a model trained for one instance is not silently reused."""
        visualizer = object.__new__(TSPVisualizer)
        visualizer.current_instance = TSPInstance(
            "old", np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        )
        new_instance = TSPInstance(
            "new", np.array([[0, 0], [2, 0], [0, 2]], dtype=float)
        )
        visualizer.ml_model = object()
        reset_calls = []
        visualizer._reset = lambda: reset_calls.append(True)
        visualizer._update_display = lambda: None

        visualizer._set_current_instance(new_instance)

        self.assertIs(visualizer.current_instance, new_instance)
        self.assertIsNone(visualizer.ml_model)
        self.assertEqual(reset_calls, [True])

    def test_ui_training_rejects_large_instance(self):
        """Test that synchronous training cannot freeze the UI on large input."""
        visualizer = object.__new__(TSPVisualizer)
        coords = np.column_stack((np.arange(16), np.zeros(16)))
        visualizer.current_instance = TSPInstance("large", coords)
        visualizer._reset = lambda: self.fail("large training should not start")

        with patch("visualization.ui.messagebox.showwarning") as showwarning:
            visualizer._train_ml_model()

        showwarning.assert_called_once()

    def test_update_display_discards_path_from_previous_instance(self):
        """Test that switching to a smaller instance cannot index an old path."""
        visualizer = object.__new__(TSPVisualizer)
        visualizer.current_instance = TSPInstance(
            "small", np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        )
        visualizer.current_path = [0, 3, 0]
        visualizer.ax = _Axes()
        visualizer.canvas = _Canvas()

        visualizer._update_display()

        self.assertIsNone(visualizer.current_path)

    def test_update_display_discards_non_integer_path(self):
        """Test that malformed city identifiers cannot index coordinates."""
        visualizer = object.__new__(TSPVisualizer)
        visualizer.current_instance = TSPInstance(
            "small", np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        )
        visualizer.current_path = [0, 1.5, 0]
        visualizer.ax = _Axes()
        visualizer.canvas = _Canvas()

        visualizer._update_display()

        self.assertIsNone(visualizer.current_path)

    def test_stale_path_animation_is_ignored_after_instance_switch(self):
        """Test that an already queued old animation cannot draw a new instance."""
        visualizer = object.__new__(TSPVisualizer)
        visualizer._search_run_id = 2
        visualizer.current_instance = TSPInstance(
            "small", np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        )

        visualizer._animate_path([0, 3, 0], animation_run_id=1)

    def test_path_animation_discards_invalid_path(self):
        """Test defensive validation inside the animation callback itself."""
        visualizer = object.__new__(TSPVisualizer)
        visualizer._search_run_id = 1
        visualizer.current_instance = TSPInstance(
            "small", np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        )
        visualizer.current_path = [0, 3, 0]
        visualizer.animation_id = "pending"
        visualizer.status_label = _Label()
        visualizer.ax = _Axes()
        visualizer.canvas = _Canvas()

        visualizer._animate_path(visualizer.current_path, animation_run_id=1)

        self.assertIsNone(visualizer.current_path)
        self.assertIsNone(visualizer.animation_id)
        self.assertIn("Path discarded", visualizer.status_label.text)


if __name__ == '__main__':
    unittest.main()
