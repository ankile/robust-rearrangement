import subprocess
import os
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--up", "-u", action="store_true")
    parser.add_argument("--down", "-d", action="store_true")
    parser.add_argument("--confirm", "-y", action="store_true")
    parser.add_argument("--subfolder", "-s", type=str, default="")
    args = parser.parse_args()

    assert args.up ^ args.down, "Must specify either --up or --down"

    aws_command = os.environ.get("AWS_COMMAND", "aws")
    command = [aws_command, "s3", "sync"]
    local = Path(os.environ.get("FURNITURE_DATA_DIR", "data")) / args.subfolder
    remote = f"s3://furniture-diffusion/data/{args.subfolder}"

    if args.up:
        command += [str(local), remote]
    else:
        command += [remote, str(local)]

    if args.confirm:
        subprocess.run(command)
    else:
        subprocess.run(command + ["--dryrun"])

        confirm = input("Confirm? [y/N] ")
        if confirm == "y":
            subprocess.run(command)
        else:
            print("Aborting")


if __name__ == "__main__":
    main()
