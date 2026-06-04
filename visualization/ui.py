"""
TSP Visualization UI Module

Features:
- Load custom TSP instances
- Select algorithm (BFS, UCS, A*, Learning A*)
- Real-time search tree visualization (color-coded expansion order)
- Final path animation
- Adjustable lambda parameter for Learning A*
- Performance optimization: batch expansions and draw the first 1000 nodes

Reference notes:
- Tkinter provides the desktop UI framework.
- Matplotlib provides embedded plotting and colormaps.
- Full library and project reuse notes are in REFERENCES.md.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Dict, Optional, Tuple

from data.data_processor import TSPInstance
from search.bfs import BFSSolver
from search.ucs import UCSSolver
from search.astar import AStarSolver


class TSPVisualizer:
    """
    TSP Visualizer
    
    Interactive UI implementation using Tkinter + Matplotlib
    """

    ALGORITHMS = {
        'BFS': {
            'class': BFSSolver, 'max_n': 10, 'color': '#3498db',
            'data_path': 'tsp_small_instances.csv'
        },
        'UCS': {
            'class': UCSSolver, 'max_n': 12, 'color': '#2ecc71',
            'data_path': 'tsp_small_instances.csv'
        },
        'A*': {
            'class': AStarSolver, 'max_n': 25, 'color': '#e74c3c',
            'data_path': 'tsp_instances_dataset.csv'
        },
        'Learning A*': {
            'class': None, 'max_n': 15, 'color': '#9b59b6',
            'data_path': 'tsp_small_instances.csv'
        }
    }
    MAX_UI_TRAINING_N = 15
    MAX_DRAWN_SEARCH_NODES = 1000
    APP_BG = '#eef3f8'
    PANEL_BG = '#ffffff'
    TEXT_COLOR = '#1f2937'
    MUTED_TEXT_COLOR = '#64748b'
    PRIMARY_COLOR = '#2563eb'
    PRIMARY_HOVER_COLOR = '#1d4ed8'
    PRIMARY_PRESSED_COLOR = '#1e40af'
    BORDER_COLOR = '#d7e1ec'
    CONTROL_BG = '#f8fafc'
    CONTROL_HOVER_BG = '#eef4fb'
    CONTROL_PRESSED_BG = '#e2e8f0'
    ML_COLOR = '#7c3aed'
    ML_HOVER_COLOR = '#6d28d9'
    ML_PRESSED_COLOR = '#5b21b6'
    PATH_COLOR = '#ef4444'
    ANIMATION_PATH_COLOR = '#16a34a'
    NODE_COLOR = '#3b82f6'

    def __init__(self, root: tk.Tk = None):
        """
        Initialize visualizer.
        
        Args:
            root: Optional Tkinter root window
        """
        self.root = root if root else tk.Tk()
        self.root.title("TSP Search Visualizer")
        self.root.geometry("1200x800")
        self.root.minsize(980, 680)
        self.root.configure(background=self.APP_BG)
        self.root.attributes('-topmost', True)  # Keep window on top
        self.root.update_idletasks()  # Update geometry
        
        # Center window on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 1200) // 2
        y = (screen_height - 800) // 2
        self.root.geometry(f"1200x800+{x}+{y}")
        self.root.attributes('-topmost', False)  # Allow other windows on top
        
        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.current_instance: Optional[TSPInstance] = None
        self.current_path: Optional[List[int]] = None
        self.search_tree: List[Dict] = []
        self.displayed_nodes = 0
        self.animation_id = None
        self.search_after_id = None
        self._search_run_id = 0
        self.is_paused = False
        self.ml_model = None
        
        self.loaded_instances: List[TSPInstance] = []
        self.filtered_instances: List[TSPInstance] = []
        self.selected_instance_name = tk.StringVar(value="")

        self.selected_algorithm = tk.StringVar(value='A*')
        self.lambda_value = tk.DoubleVar(value=0.5)
        self.lambda_display = tk.StringVar(value=f"{self.lambda_value.get():.2f}")
        self.lambda_value.trace_add("write", self._update_lambda_display)
        self.speed_value = tk.IntVar(value=50)

        self._configure_styles()
        self._create_widgets()
        self._create_menu()
        
        # Load default instances from CSV
        self._load_default_instances()

    def _configure_styles(self):
        """Configure a light visual theme without adding runtime overhead."""
        self.style = ttk.Style(self.root)
        if 'clam' in self.style.theme_names():
            self.style.theme_use('clam')

        self.style.configure('App.TFrame', background=self.APP_BG)
        self.style.configure('Header.TFrame', background=self.APP_BG)
        self.style.configure('Toolbar.TFrame', background=self.PANEL_BG)
        self.style.configure(
            'Card.TFrame', background=self.PANEL_BG, borderwidth=1,
            relief='solid', bordercolor=self.BORDER_COLOR
        )
        self.style.configure(
            'Title.TLabel', background=self.APP_BG, foreground=self.TEXT_COLOR,
            font=('Segoe UI', 18, 'bold')
        )
        self.style.configure(
            'Subtitle.TLabel', background=self.APP_BG,
            foreground=self.MUTED_TEXT_COLOR, font=('Segoe UI', 10)
        )
        self.style.configure(
            'Toolbar.TLabel', background=self.PANEL_BG,
            foreground=self.TEXT_COLOR, font=('Segoe UI', 11, 'bold')
        )
        self.style.configure(
            'Value.TLabel', background=self.PANEL_BG,
            foreground=self.PRIMARY_COLOR, font=('Segoe UI', 11, 'bold')
        )
        self.style.configure(
            'Toolbar.TButton', font=('Segoe UI', 10, 'bold'), padding=(12, 7),
            foreground='#334155', background=self.CONTROL_BG,
            bordercolor=self.BORDER_COLOR, lightcolor=self.BORDER_COLOR,
            darkcolor=self.BORDER_COLOR, borderwidth=1, relief='solid',
            focuscolor=self.BORDER_COLOR
        )
        self.style.map(
            'Toolbar.TButton',
            background=[
                ('pressed', self.CONTROL_PRESSED_BG),
                ('active', self.CONTROL_HOVER_BG)
            ],
            bordercolor=[
                ('pressed', '#94a3b8'),
                ('active', '#a8bbd0')
            ],
            foreground=[('disabled', '#94a3b8')]
        )
        self.style.configure(
            'Accent.TButton', font=('Segoe UI', 10, 'bold'),
            padding=(14, 7), foreground='#ffffff', background=self.PRIMARY_COLOR,
            bordercolor=self.PRIMARY_COLOR, lightcolor=self.PRIMARY_COLOR,
            darkcolor=self.PRIMARY_COLOR, borderwidth=1, relief='solid',
            focuscolor=self.PRIMARY_COLOR
        )
        self.style.map(
            'Accent.TButton',
            background=[
                ('pressed', self.PRIMARY_PRESSED_COLOR),
                ('active', self.PRIMARY_HOVER_COLOR)
            ],
            bordercolor=[
                ('pressed', self.PRIMARY_PRESSED_COLOR),
                ('active', self.PRIMARY_HOVER_COLOR)
            ]
        )
        self.style.configure(
            'ML.TButton', font=('Segoe UI', 10, 'bold'), padding=(12, 7),
            foreground='#ffffff', background=self.ML_COLOR,
            bordercolor=self.ML_COLOR, lightcolor=self.ML_COLOR,
            darkcolor=self.ML_COLOR, borderwidth=1, relief='solid',
            focuscolor=self.ML_COLOR
        )
        self.style.map(
            'ML.TButton',
            background=[
                ('pressed', self.ML_PRESSED_COLOR),
                ('active', self.ML_HOVER_COLOR)
            ],
            bordercolor=[
                ('pressed', self.ML_PRESSED_COLOR),
                ('active', self.ML_HOVER_COLOR)
            ]
        )
        self.style.configure(
            'Status.TLabel', background='#e8f0fb',
            foreground='#334155', font=('Segoe UI', 10), padding=(14, 8),
            borderwidth=1, relief='solid', bordercolor='#cbdbea'
        )
        self.style.configure(
            'TCombobox', font=('Segoe UI', 10), padding=5,
            fieldbackground=self.CONTROL_BG, background=self.CONTROL_BG,
            bordercolor=self.BORDER_COLOR, lightcolor=self.BORDER_COLOR,
            darkcolor=self.BORDER_COLOR, arrowcolor=self.TEXT_COLOR
        )
        self.style.map(
            'TCombobox',
            fieldbackground=[('readonly', self.CONTROL_BG)],
            selectbackground=[('readonly', self.CONTROL_BG)],
            selectforeground=[('readonly', self.TEXT_COLOR)],
            bordercolor=[('focus', self.PRIMARY_COLOR)]
        )
        self.style.configure(
            'Horizontal.TScale', background=self.PANEL_BG,
            troughcolor='#dbe5f0', bordercolor=self.BORDER_COLOR
        )

    def _create_widgets(self):
        """Create UI widgets"""
        app_frame = ttk.Frame(self.root, style='App.TFrame')
        app_frame.pack(fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(app_frame, style='Header.TFrame')
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=18, pady=(14, 8))
        ttk.Label(
            header_frame, text="TSP Search Visualizer", style='Title.TLabel'
        ).pack(anchor=tk.W)
        ttk.Label(
            header_frame,
            text="Explore routes with BFS, UCS, A*, and Learning A*",
            style='Subtitle.TLabel'
        ).pack(anchor=tk.W, pady=(2, 0))

        control_frame = ttk.Frame(app_frame, style='Card.TFrame', padding=(16, 14))
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=18, pady=(0, 10))

        settings_frame = ttk.Frame(control_frame, style='Toolbar.TFrame')
        settings_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(settings_frame, text="Algorithm", style='Toolbar.TLabel').grid(
            row=0, column=0, padx=(0, 7), pady=4, sticky=tk.W
        )
        algo_combo = ttk.Combobox(settings_frame, textvariable=self.selected_algorithm,
                                  values=list(self.ALGORITHMS.keys()), state='readonly',
                                  width=12)
        algo_combo.grid(row=0, column=1, padx=(0, 12), pady=4)
        algo_combo.bind('<<ComboboxSelected>>', self._on_algorithm_change)

        self.lambda_label = ttk.Label(
            settings_frame, text="Lambda", style='Toolbar.TLabel'
        )
        self.lambda_label.grid(
            row=0, column=2, padx=(0, 7), pady=4, sticky=tk.W
        )
        self.lambda_scale = ttk.Scale(
            settings_frame, from_=0.0, to=1.0,
            variable=self.lambda_value, orient=tk.HORIZONTAL, length=100
        )
        self.lambda_scale.grid(row=0, column=3, padx=(0, 6), pady=4)
        self.lambda_value_label = ttk.Label(
            settings_frame, textvariable=self.lambda_display,
            style='Value.TLabel', width=4
        )
        self.lambda_value_label.grid(row=0, column=4, padx=(0, 12), pady=4)
        self.lambda_widgets = (
            self.lambda_label, self.lambda_scale, self.lambda_value_label
        )

        ttk.Label(settings_frame, text="Speed", style='Toolbar.TLabel').grid(
            row=0, column=5, padx=(0, 7), pady=4, sticky=tk.W
        )
        speed_scale = ttk.Scale(settings_frame, from_=1, to=100,
                                variable=self.speed_value, orient=tk.HORIZONTAL, length=100)
        speed_scale.grid(row=0, column=6, padx=(0, 6), pady=4)
        ttk.Label(
            settings_frame, textvariable=self.speed_value,
            style='Value.TLabel', width=3
        ).grid(row=0, column=7, padx=(0, 12), pady=4)

        ttk.Label(settings_frame, text="Instance", style='Toolbar.TLabel').grid(
            row=0, column=8, padx=(0, 7), pady=4, sticky=tk.W
        )
        self.instance_combo = ttk.Combobox(
            settings_frame, textvariable=self.selected_instance_name,
            state='readonly', width=22
        )
        self.instance_combo.grid(row=0, column=9, pady=4, sticky=tk.EW)
        self.instance_combo.bind('<<ComboboxSelected>>', self._on_instance_select)
        settings_frame.columnconfigure(9, weight=1)

        btn_frame = ttk.Frame(control_frame, style='Toolbar.TFrame')
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=(9, 0))

        ttk.Button(
            btn_frame, text="Start Search", command=self._start_search,
            style='Accent.TButton'
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            btn_frame, text="Pause / Resume", command=self._pause_search,
            style='Toolbar.TButton'
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            btn_frame, text="Show Path", command=self._show_final_path,
            style='Toolbar.TButton'
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            btn_frame, text="Reset", command=self._reset,
            style='Toolbar.TButton'
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            btn_frame, text="Load Instance", command=self._load_instance,
            style='Toolbar.TButton'
        ).pack(side=tk.RIGHT, padx=(6, 0))
        self.train_model_button = ttk.Button(
            btn_frame, text="Train ML Model", command=self._train_ml_model,
            style='ML.TButton'
        )
        self.train_model_button.pack(side=tk.RIGHT, padx=3)
        self._update_learning_controls_visibility()

        self.status_label = ttk.Label(
            app_frame, text="Ready", style='Status.TLabel', anchor=tk.W
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=18, pady=(0, 12))

        self.canvas_frame = ttk.Frame(app_frame, style='Card.TFrame', padding=8)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 10))

        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8), facecolor=self.PANEL_BG)
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.08)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(highlightthickness=0, background=self.PANEL_BG)
        canvas_widget.pack(fill=tk.BOTH, expand=True)

    def _create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load TSP Instance", command=self._load_instance)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _update_lambda_display(self, *_args):
        """Keep the Lambda label compact while preserving its full value."""
        self.lambda_display.set(f"{self.lambda_value.get():.2f}")

    def _update_learning_controls_visibility(self):
        """Show ML-specific controls only when Learning A* is selected."""
        if self.selected_algorithm.get() == 'Learning A*':
            for widget in self.lambda_widgets:
                widget.grid()
            self.train_model_button.pack(side=tk.RIGHT, padx=3)
        else:
            for widget in self.lambda_widgets:
                widget.grid_remove()
            self.train_model_button.pack_forget()

    def _load_default_instances(self):
        """Load default TSP instances from CSV file based on selected algorithm"""
        algo_name = self.selected_algorithm.get()
        self._load_instances_for_algorithm(algo_name)

    def _load_instances_for_algorithm(self, algo_name):
        """Load appropriate dataset based on algorithm's city limit"""
        from data.data_processor import load_tsp_instances
        
        algo_info = self.ALGORITHMS[algo_name]
        max_n = algo_info['max_n']
        data_path = algo_info['data_path']
        
        try:
            self.loaded_instances = load_tsp_instances(num_instances=100, data_path=data_path)
            self._filter_instances_by_algorithm(algo_name)
            self.status_label.config(text=f"Algorithm: {algo_name} (max {max_n} cities)")
        except Exception as e:
            self.status_label.config(text=f"Warning: Could not load instances: {str(e)}")

    def _filter_instances_by_algorithm(self, algo_name=None):
        """Filter instances based on selected algorithm's city limit"""
        if algo_name is None:
            algo_name = self.selected_algorithm.get()
        
        max_n = self.ALGORITHMS[algo_name]['max_n']
        
        self.filtered_instances = [
            inst for inst in self.loaded_instances 
            if inst.n <= max_n
        ]
        
        instance_names = [f"{inst.name} ({inst.n} cities)" for inst in self.filtered_instances]
        self.instance_combo['values'] = instance_names
        
        if instance_names:
            self.selected_instance_name.set(instance_names[0])
            self._set_current_instance(self.filtered_instances[0])
        else:
            self._set_current_instance(None)
            self.selected_instance_name.set("")

    def _on_algorithm_change(self, event):
        """Handle algorithm selection change"""
        algo_name = self.selected_algorithm.get()
        self._update_learning_controls_visibility()
        self._load_instances_for_algorithm(algo_name)

    def _on_instance_select(self, event):
        """Handle instance selection change"""
        selected_name = self.selected_instance_name.get()
        for inst in self.filtered_instances:
            display_name = f"{inst.name} ({inst.n} cities)"
            if display_name == selected_name:
                self._set_current_instance(inst)
                self.status_label.config(text=f"Loaded: {inst.name} ({inst.n} cities)")
                break

    def _load_instance(self):
        """Load TSP instance from file"""
        filename = filedialog.askopenfilename(
            title="Select TSP Instance",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )

        if not filename:
            return

        try:
            from data.data_processor import load_tsp_instances
            instances = load_tsp_instances(data_path=filename)
            if instances:
                self.loaded_instances = instances
                self._filter_instances_by_algorithm()
                if self.current_instance:
                    self.status_label.config(
                        text=(
                            f"Loaded {len(self.filtered_instances)} instance(s) "
                            f"from file; selected: {self.current_instance.name} "
                            f"({self.current_instance.n} cities)"
                        )
                    )
                else:
                    algo_info = self.ALGORITHMS[self.selected_algorithm.get()]
                    messagebox.showwarning(
                        "Warning",
                        f"No instances in this file match {self.selected_algorithm.get()} "
                        f"(N <= {algo_info['max_n']})."
                    )
                    self.status_label.config(text="No compatible instances loaded")
            else:
                messagebox.showerror("Error", "Failed to load instance")
        except Exception as e:
            messagebox.showerror("Error", f"Load failed: {str(e)}")

    def _update_display(self):
        """Update visualization display"""
        if not self.current_instance:
            return

        self.ax.clear()

        coords = self.current_instance.coords

        self.ax.scatter(
            coords[:, 0], coords[:, 1], c=self.NODE_COLOR, s=110,
            edgecolors='#ffffff', linewidths=1.2, zorder=5
        )

        for i, (x, y) in enumerate(coords):
            self.ax.annotate(str(i), (x, y), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=10)

        if self.current_path and len(self.current_path) > 1:
            if self._path_matches_instance(self.current_path, len(coords)):
                path_coords = coords[self.current_path]
                self.ax.plot(
                    path_coords[:, 0], path_coords[:, 1],
                    color=self.PATH_COLOR, linewidth=2.4, zorder=3
                )
            else:
                self.current_path = None

        self._style_axes(f"TSP Instance: {self.current_instance.name}")
        self.canvas.draw_idle()

    def _style_axes(self, title: str):
        """Apply consistent plot styling after axes are cleared."""
        self.ax.set_facecolor('#f8fafc')
        self.ax.set_title(
            title, fontsize=14, fontweight='semibold',
            color=self.TEXT_COLOR, pad=12
        )
        self.ax.grid(True, color='#dbe4ee', linewidth=0.8, alpha=0.8)
        self.ax.tick_params(colors=self.MUTED_TEXT_COLOR, labelsize=9)

    @staticmethod
    def _search_animation_settings(speed: int) -> Tuple[int, int]:
        """Return nodes processed per frame and the delay between frames."""
        bounded_speed = max(1, min(100, int(speed)))
        batch_size = max(1, bounded_speed * bounded_speed // 100)
        delay = max(1, (100 - bounded_speed) * 2)
        return batch_size, delay

    def _start_search(self):
        """Start search algorithm with real-time visualization"""
        if not self.current_instance:
            messagebox.showwarning("Warning", "Please load a TSP instance first")
            return

        algo_name = self.selected_algorithm.get()
        algo_info = self.ALGORITHMS[algo_name]

        if algo_name == 'Learning A*' and self.ml_model is None:
            messagebox.showwarning(
                "Warning",
                "Learning A* requires a trained model. Click Train ML Model first."
            )
            return

        if self.current_instance.n > algo_info['max_n']:
            messagebox.showwarning("Warning", f"{algo_name} only supports N <= {algo_info['max_n']}")
            return

        self._reset()
        search_run_id = self._search_run_id
        self.status_label.config(text=f"Running {algo_name}...")

        dist = self.current_instance.dist_matrix
        coords = self.current_instance.coords

        def _record_node(node_info):
            """Retain enough expansion history to draw the search progress."""
            self.displayed_nodes += 1
            if len(self.search_tree) < self.MAX_DRAWN_SEARCH_NODES:
                self.search_tree.append(node_info)

        def _render_search_progress():
            """Draw one animation frame after a batch of expansions."""
            self.ax.clear()
            self.ax.scatter(
                coords[:, 0], coords[:, 1], c=self.NODE_COLOR, s=110,
                edgecolors='#ffffff', linewidths=1.2, zorder=5
            )

            for i, (x, y) in enumerate(coords):
                self.ax.annotate(str(i), (x, y), textcoords="offset points",
                               xytext=(0, 10), ha='center', fontsize=10)

            if len(self.search_tree) > 1:
                max_nodes = len(self.search_tree)
                colors = plt.cm.plasma(np.linspace(0, 0.9, max_nodes))

                for idx in range(1, max_nodes):
                    node = self.search_tree[idx]
                    parent_idx = node['parent']

                    if parent_idx >= 0 and parent_idx < idx:
                        parent_node = self.search_tree[parent_idx]
                        x1, y1 = coords[parent_node['state'].current_city]
                        x2, y2 = coords[node['state'].current_city]

                        self.ax.plot([x1, x2], [y1, y2], '-',
                                   color=colors[idx], alpha=0.6, linewidth=1.5)

            self._style_axes(f"{algo_name} Search - Expanded Nodes: {self.displayed_nodes}")
            self.canvas.draw_idle()

        def _step_search(solver, search_gen, algo_name):
            """Single step of search with pause support"""
            if search_run_id != self._search_run_id:
                return
            if self.is_paused:
                # Schedule next check
                self.search_after_id = self.root.after(
                    50, lambda: _step_search(solver, search_gen, algo_name)
                )
                return
            
            batch_size, delay = self._search_animation_settings(self.speed_value.get())
            search_complete = False

            for _ in range(batch_size):
                try:
                    node_info = next(search_gen)
                except StopIteration:
                    search_complete = True
                    break

                _record_node(node_info)

                if node_info.get('is_complete', False):
                    search_complete = True
                    break

            _render_search_progress()

            if search_complete:
                result = solver.get_result()
                self.search_after_id = None
                self._on_search_complete(result, algo_name)
                return

            # Schedule the next animation frame.
            self.search_after_id = self.root.after(
                delay, lambda: _step_search(solver, search_gen, algo_name)
            )

        try:
            if algo_name == 'BFS':
                solver = BFSSolver(dist)
                solver.prepare_for_iteration()
                search_gen = solver.search_generator()
                _step_search(solver, search_gen, algo_name)
            elif algo_name == 'UCS':
                solver = UCSSolver(dist)
                solver.prepare_for_iteration()
                search_gen = solver.search_generator()
                _step_search(solver, search_gen, algo_name)
            elif algo_name == 'A*':
                solver = AStarSolver(dist)
                solver.prepare_for_iteration()
                search_gen = solver.search_generator()
                _step_search(solver, search_gen, algo_name)
            elif algo_name == 'Learning A*':
                from ml.learning_astar import LearningAStar
                solver = LearningAStar(dist, model=self.ml_model,
                                     lambda_param=self.lambda_value.get())
                solver.prepare_for_iteration()
                search_gen = solver.search_generator()
                _step_search(solver, search_gen, algo_name)
        except Exception as e:
            messagebox.showerror("Error", f"Search failed: {str(e)}")
            self.status_label.config(text=f"Error: {str(e)}")

    def _on_search_complete(self, result: Dict, algo_name: str):
        """Search completion callback"""
        if result['success']:
            self.current_path = result['path']
            self._update_display()
            self.status_label.config(
                text=f"{algo_name} completed: Cost={result['cost']:.2f}, "
                     f"Nodes={result['nodes_expanded']}, "
                     f"Time={self._format_elapsed_time(result['time'])}"
            )
        else:
            messagebox.showinfo("Result", f"{algo_name} failed to find solution")

    @staticmethod
    def _format_elapsed_time(seconds: float) -> str:
        """Format very short runtimes without rounding them down to 0."""
        if seconds >= 1:
            return f"{seconds:.3f}s"
        if seconds >= 0.001:
            return f"{seconds * 1000:.3f}ms"
        return f"{seconds * 1_000_000:.1f}us"

    def _pause_search(self):
        """Toggle pause/resume search animation"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.status_label.config(text="Paused - Click Pause again to resume")
        else:
            self.status_label.config(text="Resumed")

    def _reset(self):
        """Reset visualization"""
        self._search_run_id += 1
        self.is_paused = False
        if self.search_after_id is not None:
            try:
                self.root.after_cancel(self.search_after_id)
            except tk.TclError:
                pass
            self.search_after_id = None
        if self.animation_id is not None:
            try:
                self.root.after_cancel(self.animation_id)
            except tk.TclError:
                pass
            self.animation_id = None
        self.current_path = None
        self.search_tree = []
        self.displayed_nodes = 0
        if self.current_instance:
            self._update_display()
        self.status_label.config(text="Reset")

    def _set_current_instance(self, instance: Optional[TSPInstance]):
        """Switch instances and discard state that belongs to the old one."""
        self._reset()
        self.current_instance = instance
        self.ml_model = None
        if self.current_instance:
            self._update_display()

    @staticmethod
    def _path_matches_instance(path: List[int], n_cities: int) -> bool:
        """Return whether a path can safely index the current coordinates."""
        return all(
            isinstance(city, (int, np.integer)) and 0 <= city < n_cities
            for city in path
        )

    def _show_final_path(self):
        """Show final path animation"""
        if not self.current_path:
            messagebox.showinfo("Info", "No path to display")
            return

        if self.animation_id is not None:
            try:
                self.root.after_cancel(self.animation_id)
            except tk.TclError:
                pass
        self.displayed_nodes = 0
        animation_run_id = self._search_run_id
        self._animate_path(self.current_path, animation_run_id)

    def set_ml_model(self, model):
        """Configure the trained model used by Learning A*."""
        self.ml_model = model

    def _train_ml_model(self):
        """Train a Random Forest model from the currently loaded instance."""
        if not self.current_instance:
            messagebox.showwarning("Warning", "Please load a TSP instance first")
            return

        if self.current_instance.n > self.MAX_UI_TRAINING_N:
            messagebox.showwarning(
                "Warning",
                f"UI model training supports at most {self.MAX_UI_TRAINING_N} cities. "
                "Select a smaller instance first."
            )
            return

        try:
            from ml.model import TSPMLModel
            from ml.pseudo_labels import generate_pseudo_labels_for_instance

            self._reset()
            self.status_label.config(text="Training ML model...")
            self.root.update_idletasks()
            X, y = generate_pseudo_labels_for_instance(self.current_instance)
            model = TSPMLModel("rf")
            metrics = model.train(X, y)
            self.set_ml_model(model)
            self.status_label.config(
                text=f"ML model trained: Samples={len(y)}, Accuracy={metrics['accuracy']:.3f}"
            )
        except Exception as exc:
            messagebox.showerror("Error", f"ML model training failed: {exc}")
            self.status_label.config(text=f"ML model training failed: {exc}")

    def _animate_path(self, path: List[int], animation_run_id: int = None):
        """Animate path drawing"""
        if animation_run_id is None:
            animation_run_id = self._search_run_id
        if animation_run_id != self._search_run_id or not self.current_instance:
            return

        coords = self.current_instance.coords
        if not self._path_matches_instance(path, len(coords)):
            self.current_path = None
            self.animation_id = None
            self.status_label.config(text="Path discarded: it does not match the current instance")
            self._update_display()
            return

        self.ax.clear()
        self.ax.scatter(
            coords[:, 0], coords[:, 1], c=self.NODE_COLOR, s=110,
            edgecolors='#ffffff', linewidths=1.2, zorder=5
        )

        for i, (x, y) in enumerate(coords):
            self.ax.annotate(str(i), (x, y), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=10)

        path_coords = coords[path[:self.displayed_nodes + 1]]
        if len(path_coords) > 0:
            self.ax.plot(
                path_coords[:, 0], path_coords[:, 1],
                color=self.ANIMATION_PATH_COLOR, linewidth=2.6, zorder=3
            )

        self._style_axes(f"Route Animation: {self.current_instance.name}")
        self.canvas.draw_idle()

        if self.displayed_nodes < len(path) - 1:
            self.displayed_nodes += 1
            delay = int(110 - self.speed_value.get())
            self.animation_id = self.root.after(
                delay, lambda: self._animate_path(path, animation_run_id)
            )
        else:
            self.animation_id = None

    def _show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About", "TSP Search Visualizer\nSupports BFS, UCS, A*, Learning A* algorithms")

    def _on_close(self):
        """Handle window close event"""
        self._reset()
        # Clean up matplotlib figures
        plt.close('all')
        # Destroy the window
        self.root.destroy()
    
    def run(self):
        """Run the visualizer"""
        self.root.mainloop()


class SimpleVisualizer:
    """
    Simple Visualizer (for non-GUI environments)
    
    Uses matplotlib to generate static images
    """

    def __init__(self, instance: TSPInstance):
        """
        Initialize simple visualizer.
        
        Args:
            instance: TSPInstance to visualize
        """
        self.instance = instance

    def plot_instance(self, path: List[int] = None,
                     save_path: str = None):
        """
        Plot TSP instance.
        
        Args:
            path: Optional path to highlight
            save_path: Optional path to save image
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        coords = self.instance.coords

        ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=100, zorder=5)

        for i, (x, y) in enumerate(coords):
            ax.annotate(str(i), (x, y), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=10)

        if path and len(path) > 1:
            path_coords = coords[path]
            ax.plot(path_coords[:, 0], path_coords[:, 1], 'r-', linewidth=2, zorder=3)
            ax.scatter([coords[path[0]][0]], [coords[path[0]][1]],
                      c='green', s=150, marker='*', zorder=6)

        ax.set_title(f"TSP: {self.instance.name}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()

    def plot_search_tree(self, search_tree: List[Dict],
                       max_display: int = 1000,
                       save_path: str = None):
        """
        Plot search tree.
        
        Args:
            search_tree: List of search tree nodes
            max_display: Maximum number of nodes to display
            save_path: Optional path to save image
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        coords = self.instance.coords

        ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=100, zorder=5)

        for i, (x, y) in enumerate(coords):
            ax.annotate(str(i), (x, y), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=10)

        colors = plt.cm.viridis(np.linspace(0, 1, min(len(search_tree), max_display)))

        for idx, node in enumerate(search_tree[:max_display]):
            state = node['state']
            parent_idx = node['parent']

            if parent_idx >= 0 and parent_idx < idx:
                parent_state = search_tree[parent_idx]['state']
                x1, y1 = coords[parent_state.current_city]
                x2, y2 = coords[state.current_city]
                ax.plot([x1, x2], [y1, y2], '-',
                       color=colors[idx], alpha=0.5, linewidth=1)

        ax.set_title(f"Search Tree ({len(search_tree)} nodes)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = TSPVisualizer(root)
        app.run()
    except Exception as e:
        print(f"Tkinter unavailable, using simple visualizer: {e}")
        print("Example usage:")
        print("  from visualization import SimpleVisualizer")
        print("  viz = SimpleVisualizer(instance)")
        print("  viz.plot_instance(path)")
