from dataclasses import dataclass


@dataclass
class Params:
    """PUMA params for ladder step 4b: one self-intersecting capricorn demo, 2nd order, 2D."""

    dataset_name: str = "ladder4_capricorn"
    results_path: str = "results/ladder4_2nd_order_2D/ladder4_capricorn/"
    multi_motion: bool = False
    selected_primitives_ids: str = "0"
    manifold_dimensions: int = 2
    saturate_out_of_boundaries_transitions: bool = True
    dynamical_system_order: int = 2
    space: str = "euclidean"

    latent_space_dim: int = 300
    neurons_hidden_layers: int = 300
    batch_size: int = 250
    learning_rate: float = 0.0001245
    weight_decay: float = 0.0

    triplet_type: str = "spherical"
    imitation_loss_weight: float = 1
    stabilization_loss_weight: float = 0.447267
    boundary_loss_weight: float = 0
    imitation_window_size: int = 15
    stabilization_window_size: int = 12
    triplet_margin: float = 2.9194e-7
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
    evaluation_interval: int = 1500
    quanti_eval: bool = True
    quali_eval: bool = True
    diffeo_quanti_eval: bool = True # False
    diffeo_quali_eval: bool = True # False
    ignore_n_spurious: bool = False
    fixed_point_iteration_thr = 10
    density: int = 25
    simulated_trajectory_length: int = 2000
    evaluation_samples_length: int = 100
    show_plot: bool = False

    gamma_objective = 3.5
    optuna_n_trials = 1000
    length_dataset = 1

    def __init__(self, results_base_directory):
        self.results_path = results_base_directory + self.results_path
