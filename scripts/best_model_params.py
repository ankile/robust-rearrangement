import os
import wandb
import pandas as pd
from wandb.apis.public import Run


api = wandb.Api()

from ipdb import set_trace as bp

# Specify the path to the folder containing the CSV files
csv_path = "/data/scratch/ankile/robust-rearrangement/outputs/ol-state-dr-high-1/8o5iet5y_results.csv"

# Initialize variables to store the best weight, its success rate, and the corresponding run ID

project_id = csv_path.split("/")[-2]
run_id = csv_path.split("/")[-1].split("_")[0]

# Headers: wt_type,success_rate
df = pd.read_csv(csv_path)

# Find the best weight and its success rate
best_weight = df.loc[df["success_rate"].idxmax()]["wt_type"].astype(int).astype(str)


print(f"Best weight: {best_weight}")

# Checkpoint path example: models/actor_chkpt_1099.pt

# Download the weights from the files
run: Run = api.run(f"{project_id}/{run_id}")


for file in run.files():
    print(file.name)
    if file.name.endswith("actor_chkpt_best_success_rate.pt"):
        file.delete()

run.update()

# Download the weights from the files
for file in run.files():
    print(file.name)
    if file.name.endswith(f"_{best_weight}.pt"):
        file.download(replace=True, exist_ok=True)
        print(f"Downloaded {file.name}")
        break

filename = file.name

# Change name of file models/dulcet-pyramid-6/actor_chkpt_499.pt -> models/dulcet-pyramid-6/best_success_rate_499.pt
new_filename = filename.replace("actor_chkpt", "best_success_rate")

# Update the name of the downloaded file to best_success_rate.pt
os.rename(filename, new_filename)

# Upload the best weight to the current run
run.upload_file(new_filename)

run.update()
