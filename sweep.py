import io
import os
import tyro
import wandb
import jax
from dataclasses import dataclass
from contextlib import redirect_stdout

from params import Args
from utils.logging_utils import parse_config, sweep_length
from main import main


@dataclass
class RunArgs:
    config: str                     # path to config file
    id: int = 0                     # sweep id
    length: bool = False            # if true, only print the sweep length and exit
    redirect_stdout: bool = True    # capture stdout to file


if __name__ == "__main__":
    wandb.require("legacy-service")
    print("JAX devices:", jax.devices())
    print("JAX default device:", jax.devices()[0])

    rargs = tyro.cli(RunArgs)
    
    # check sweep length
    if rargs.length:
        print(f"sweep_length: {sweep_length(rargs.config)}")
        exit(0)

    args = parse_config(Args, rargs.config, rargs.id)

    print("="*70)
    print(f"Configuration:")
    print(f"  Environment type: {args.env_type}")
    print(f"  Seed: {args.seed}")
    print("="*70)

    if rargs.redirect_stdout:
        str_io = io.StringIO()
        with redirect_stdout(str_io):
            eval_dir = main(args)
        with open(os.path.join(eval_dir, "output.txt"), "w") as f:
            f.write(str_io.getvalue())
    else:
        main(args)
