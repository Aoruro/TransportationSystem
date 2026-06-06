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


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Combo:
    def __init__(self):
        self.values = []

    def __setitem__(self, key, value):
        if key == "values":
            self.values = value


class _GridWidget:
    def __init__(self):
        self.visible = True

    def grid(self):
        self.visible = True

    def grid_remove(self):
        self.visible = False


class _PackedWidget:
    def __init__(self):
        self.visible = True

    def pack(self, **_kwargs):
        self.visible = True

    def pack_forget(self):
        self.visible = False


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

    def set_facecolor(self, *_args, **_kwargs):
        pass

    def grid(self, *_args, **_kwargs):
        pass

    def tick_params(self, *_args, **_kwargs):
        pass


class _Canvas:
    def draw(self):
        pass

    def draw_idle(self):
        pass


class TestVisualizerState(unittest.TestCase):
    """UI state tests that do not require a Tk window."""

    def test_learning_astar_defaults_are_training_compatible(self):
        """Test that Learning A* examples stay within the UI training limit."""
        algo_info = TSPVisualizer.ALGORITHMS['Learning A*']
        self.assertLessEqual(algo_info['max_n'], TSPVisualizer.MAX_UI_TRAINING_N)
        self.assertEqual(algo_info['data_path'], 'tsp_small_instances.csv')

    def test_search_animation_batches_more_nodes_at_higher_speed(self):
        """Test that the speed slider reduces redraw frequency at high values."""
        self.assertEqual(TSPVisualizer._search_animation_settings(1), (1, 198))
        self.assertEqual(TSPVisualizer._search_animation_settings(50), (25, 100))
        self.assertEqual(TSPVisualizer._search_animation_settings(100), (100, 1))

    def test_short_elapsed_times_are_not_rounded_to_zero(self):
        """Test that UI status text preserves sub-second timings."""
        self.assertEqual(TSPVisualizer._format_elapsed_time(0.125), "125.000ms")
        self.assertEqual(TSPVisualizer._format_elapsed_time(0.000125), "125.0us")

    def test_speed_display_uses_at_least_two_digits(self):
        """Test that single-digit speeds do not shrink the label text."""
        self.assertEqual(TSPVisualizer._format_speed_value(1), "01")
        self.assertEqual(TSPVisualizer._format_speed_value(9), "09")
        self.assertEqual(TSPVisualizer._format_speed_value(50), "50")
        self.assertEqual(TSPVisualizer._format_speed_value(100), "100")

    def test_ml_controls_are_visible_only_for_learning_astar(self):
        """Test that regular searches do not show irrelevant ML settings."""
        visualizer = object.__new__(TSPVisualizer)
        visualizer.selected_algorithm = _Var("A*")
        visualizer.lambda_widgets = tuple(_GridWidget() for _ in range(3))
        visualizer.train_model_button = _PackedWidget()

        visualizer._update_learning_controls_visibility()
        self.assertTrue(all(not widget.visible for widget in visualizer.lambda_widgets))
        self.assertFalse(visualizer.train_model_button.visible)

        visualizer.selected_algorithm.value = "Learning A*"
        visualizer._update_learning_controls_visibility()
        self.assertTrue(all(widget.visible for widget in visualizer.lambda_widgets))
        self.assertTrue(visualizer.train_model_button.visible)

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

        with patch("visualization.ui.messagebox.showinfo") as showinfo:
            visualizer._train_ml_model()

        self.assertIsNotNone(visualizer.ml_model)
        self.assertEqual(reset_calls, [True])
        self.assertIn("ML model trained", visualizer.status_label.text)
        showinfo.assert_called_once_with(
            "Training Complete",
            "ML model trained successfully."
        )

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

    def test_load_instance_refreshes_the_instance_selector(self):
        """Test that loading a CSV replaces the selectable instance list."""
        visualizer = object.__new__(TSPVisualizer)
        visualizer.selected_algorithm = _Var("A*")
        visualizer.selected_instance_name = _Var("")
        visualizer.instance_combo = _Combo()
        visualizer.status_label = _Label()
        visualizer.current_instance = None
        visualizer.ml_model = None
        visualizer._reset = lambda: None
        visualizer._update_display = lambda: None
        instances = [
            TSPInstance("one", np.array([[0, 0], [1, 0], [0, 1]], dtype=float)),
            TSPInstance("two", np.array([[0, 0], [2, 0], [0, 2]], dtype=float)),
        ]

        with patch("visualization.ui.filedialog.askopenfilename", return_value="custom.csv"):
            with patch("data.data_processor.load_tsp_instances", return_value=instances):
                visualizer._load_instance()

        self.assertEqual(visualizer.loaded_instances, instances)
        self.assertEqual(visualizer.filtered_instances, instances)
        self.assertEqual(visualizer.instance_combo.values, ["one (3 cities)", "two (3 cities)"])
        self.assertIs(visualizer.current_instance, instances[0])
        self.assertEqual(visualizer.selected_instance_name.get(), "one (3 cities)")

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
