import os
import json
import wandb
import yaml
from dataclasses import asdict
from params import TrainArgs, EvalArgs, Args
import itertools
import hashlib


def create_sweep_args(d):
    """
    Generate all combinations of parameter sweeps from a nested dictionary definition.

    This function interprets a configuration dictionary where each key maps to a list
    of values to sweep over. Keys that contain a "+" indicate that multiple parameters
    should be swept together — i.e., their values are grouped and varied jointly
    across corresponding indices.

    The output is a list of dictionaries, each representing one unique combination
    of arguments to be used in an experimental sweep or grid search.

    Parameters
    ----------
    d : dict
        Dictionary defining the parameter sweep. Each entry should be one of:

        - `key: list` — a standard sweep over all combinations of these values.
        - `"a+b": { "a": list, "b": list }` — a grouped sweep where parameters
          `a` and `b` are varied together elementwise.

        Example:
        >>> d = {
        ...     "lr": [0.01, 0.1],
        ...     "optimizer+momentum": {"optimizer": ["sgd", "adam"], "momentum": [0.9, 0.95]},
        ... }

    Returns
    -------
    list of dict
        Each dictionary contains one complete configuration of parameters
        for a single experiment or run. For the example above:

        >>> create_sweep_args(d)
        [
            {"lr": 0.01, "optimizer": "sgd", "momentum": 0.9},
            {"lr": 0.01, "optimizer": "adam", "momentum": 0.95},
            {"lr": 0.1, "optimizer": "sgd", "momentum": 0.9},
            {"lr": 0.1, "optimizer": "adam", "momentum": 0.95},
        ]

    Notes
    -----
    - The grouped keys (with "+") ensure elementwise pairing across lists, not Cartesian products.
    - All lists used for grouped parameters must be of the same length.
    - The function expands grouped keys back into individual arguments in the output.

    See Also
    --------
    itertools.product : Used internally to compute the Cartesian product.
    Takes a diction of args to sweep over and creates the sweep.
    """
    args_to_sweep = {}
    for k in d:
        if "+" in k:
            inner_keys = list(d[k].keys())
            n_values = len(d[k][inner_keys[0]])
            args_to_sweep["+".join(inner_keys)] = [
                tuple(d[k][ik][i] for ik in inner_keys) for i in range(n_values)
            ]
        else:
            args_to_sweep[k] = d[k]

    _sweep_args = [
        {k: v[idx] for idx, k in enumerate(args_to_sweep.keys())}
        for v in itertools.product(*args_to_sweep.values())
    ]

    def parse_stuff(md):
        ret_args = {}
        for k in md:
            if "+" in k:
                inner_keys = k.split("+")
                for i, ik in enumerate(inner_keys):
                    ret_args[ik] = md[k][i]
            else:
                ret_args[k] = md[k]
        return ret_args

    return [parse_stuff(md) for md in _sweep_args]


def sweep_length(cfg_file):
    """Parse a YAML config file and determine the length."""
    with open(cfg_file, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return len(create_sweep_args(cfg_dict["sweep"]))


def parse_config(arg_constructor, cfg_file, id):
    """
    Parse a YAML configuration file and initialize a configuration object.

    This function loads a YAML configuration file, expands parameter sweeps,
    and returns a constructed argument object for the specified sweep index.

    Parameters
    ----------
    arg_constructor : callable
        Callable (class, function, or dataclass) that accepts keyword arguments
        and returns a configuration object.
    cfg_file : str
        Path to the YAML configuration file containing base and sweep definitions.
    id : int
        Index of the parameter combination to select from the sweep.

    Returns
    -------
    Any
        Configuration object initialized with parameters corresponding to `id`.

    Example
    -------
    >>> args = parse_config(Args, "configs/experiment.yaml", id=3)
    >>> args.lr
    0.1

    Notes
    -----
    - The YAML file should define a top-level key `"sweep"`.
    - Each sweep entry is processed via `create_sweep_args`.
    """
    with open(cfg_file, "r") as f:
        cfg_dict = yaml.safe_load(f)
    _sweep_args = create_sweep_args(cfg_dict["sweep"])
    args_dict = {k: v for k, v in cfg_dict.items() if k != "sweep"}
    for k, v in _sweep_args[id].items():
        args_dict[k] = v
    args = arg_constructor(**args_dict)
    return args


def setup_wandb(args: TrainArgs, run_name: str, use_wandb=False):
    if not use_wandb:
        wandb.init(mode="offline")
    else:
        if args.env_type == "RoomEnv":
            group_name = f"{args.env_type}/{args.rooms_env_name}/{args.obs_type}"
        elif args.env_type == "OGBenchEnv":
            group_name = f"{args.env_type}/{args.ogbench_env_type}/{args.ogbench_task_name}"
        else:
            group_name = args.env_type
        wandb.init(
            project=getattr(args, "project_name", "ALPS"),
            name=run_name,
            group=group_name,
            config=(
                asdict(args) if hasattr(args, "__dataclass_fields__") else vars(args)
            ),
        )


def setup_logging(args: Args):
    """
    setup logging directories

    Structure:
    results/
      env_name/
        {train_hash}/          <- Training hyperparameters
          seed-X/
            allo_model.pkl     <- Models here
            dynamics_model.pkl
            checkpoint_80/     <- Checkpoint directories (if enabled store here)
            checkpoint_90/
            checkpoint_100/
            eval_{eval_hash}/  <- Eval hyperparameters
              config.json      <- Full config
              results/         <- All evaluation outputs

    Returns
    -------
    tuple
        (model_dir, results_dir, checkpoint_dirs) where:
        - model_dir: Base directory for models (seed-X/)
        - results_dir: Where results are saved (seed-X/eval_xxx/)
        - checkpoint_dirs: Dict mapping percentage to checkpoint directory path ({100: seed-X/checkpoint_100/}).
    """
    if args.env_type == "RoomEnv":
        env_name = args.rooms_env_name
    elif args.env_type == "OGBenchEnv":
        env_name = args.ogbench_task_name
    else:
        raise ValueError(f"Unknown env_type {args.env_type}")
    
    base_dir = os.path.join(os.path.abspath(args.save_dir), env_name)
    
    # ===== TRAINING HASH =====
    hasher = hashlib.sha1()
    base_fields = {f.name for f in TrainArgs.__dataclass_fields__.values()}
    
    args_dict = asdict(args)
    train_hash_dict = dict(sorted(
        {k: v for k, v in args_dict.items() if k in base_fields and k != 'seed'}.items()
    ))
    
    hasher.update(str(train_hash_dict).encode())
    train_hash = hasher.hexdigest()
    
    train_dir = os.path.join(base_dir, f"train_{train_hash[:14]}")
    os.makedirs(train_dir, exist_ok=True)
    
    seed_dir = os.path.join(train_dir, f"seed-{args.seed}")
    os.makedirs(seed_dir, exist_ok=True)
    
    # models are saved/loaded from seed_dir
    model_dir = seed_dir
    
    # ===== EVAL HASH =====
    eval_hasher = hashlib.sha1()
    eval_fields = {f.name for f in EvalArgs.__dataclass_fields__.values()}
    eval_hash_dict = dict(sorted(
        {k: v for k, v in args_dict.items() if k in eval_fields}.items()
    ))
    
    eval_hasher.update(str(eval_hash_dict).encode())
    eval_hash = eval_hasher.hexdigest()
    
    results_dir = os.path.join(seed_dir, f"eval_{eval_hash[:14]}")
    os.makedirs(results_dir, exist_ok=True)
    
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(args), f, indent=4)
    
    # save training-only config in seed_dir (for reference)
    train_config_path = os.path.join(seed_dir, "train_config.json")
    if not os.path.exists(train_config_path):  # write once during training
        train_config = {k: v for k, v in args_dict.items() if k in base_fields}
        with open(train_config_path, "w") as f:
            json.dump(train_config, f, indent=4)

    # ===== CHECKPOINT DIRECTORIES =====
    checkpoint_dirs = {}
    for frac in args.checkpoint_fractions:
        pct = int(frac * 100)
        dir_path = os.path.join(model_dir, f"checkpoint_{pct}")
        os.makedirs(dir_path, exist_ok=True)
        checkpoint_dirs[pct] = dir_path

    # setup wandb
    wandb_name = f"{env_name}_{args.num_eigenvectors}ev_seed{args.seed}_train_{train_hash}_eval{eval_hash}"
    setup_wandb(args, wandb_name, args.use_wandb)

    return model_dir, results_dir, checkpoint_dirs
