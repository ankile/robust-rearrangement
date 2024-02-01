import argparse
from ml_collections import ConfigDict

from wandb import Api


from src.train.bc import main as train_bc_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    args = parser.parse_args()

    # Get the model file and config
    api = Api()
    run = api.run(f"robot-rearrangement/{args.run_id}")
    model_file = [f for f in run.files() if f.name.endswith(".pt")][0]
    model_path = model_file.download(exist_ok=True).name
    config = ConfigDict(run.config)
    config.wandb.continue_run_id = run.id

    # Get the epoch the model was saved at
    start_epoch = run.summary["epoch"]

    # Add the new data to the config
    config.load_checkpoint_path = model_path

    # Change any parameters here
    config.num_epochs = 200
    # config.augment_image = True
    # config.save_rollouts = False
    # config.rollout.every = -1
    config.batch_size = 64
    config.dataloader_workers = 4
    # config.num_envs = 8
    # config.data_path = "/data/scratch/ankile/furniture-data/processed/sim/one_leg/scripted/success.zarr"
    # config.actor_lr = 5e-5
    # config.rollout.loss_threshold = 1
    config.data_subset = 50

    print("Using data", config.data_path)

    train_bc_main(config, start_epoch=start_epoch)
