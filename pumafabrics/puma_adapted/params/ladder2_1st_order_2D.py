from dataclasses import dataclass


@dataclass
class Params:
    """PUMA params for ladder step 2a: one straight-line demo, 1st order, 2D."""

    dataset_name: str = "ladder2_line"
    results_path: str = "results/ladder2_1st_order_2D/ladder2_line/"
    multi_motion: bool = False
    selected_primitives_ids: str = "0"
    manifold_dimensions: int = 2
    saturate_out_of_boundaries_transitions: bool = True
    dynamical_system_order: int = 1
    space: str = "euclidean"

    latent_space_dim: int = 300
    neurons_hidden_layers: int = 300
    batch_size: int = 250
    learning_rate: float = 0.0008574
    weight_decay: float = 0.0

    triplet_type: str = "spherical"
    imitation_loss_weight: float = 1
    stabilization_loss_weight: float = 3.496
    boundary_loss_weight: float = 0
    imitation_window_size: int = 14
    stabilization_window_size: int = 14
    triplet_margin: float = 3.0122e-05
    interpolation_sigma: float = 0.8

    train: bool = True
    load_model: bool = False
    max_iterations: int = 12000

    spline_sample_type: str = "from data"
    workspace_boundaries_type: str = "from data"
    workspace_boundaries: str = "not used"
    trajectories_resample_length: int = 2000
    state_increment: float = 0.3

    save_evaluation: bool = True
    evaluation_interval: int = 3000
    quanti_eval: bool = True
    quali_eval: bool = False
    diffeo_quanti_eval: bool = False
    diffeo_quali_eval: bool = False
    ignore_n_spurious: bool = False
    fixed_point_iteration_thr = 2
    density: int = 25
    simulated_trajectory_length: int = 2000
    evaluation_samples_length: int = 100
    show_plot: bool = False

    gamma_objective = 3.5
    optuna_n_trials = 1000
    length_dataset = 1

    def __init__(self, results_base_directory):
        self.results_path = results_base_directory + self.results_path
