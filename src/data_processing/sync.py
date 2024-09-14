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
    parser.add_argument("--delete", "-D", action="store_true")
    args = parser.parse_args()

    assert args.up ^ args.down, "Must specify either --up or --down"

    aws_command = os.environ.get("AWS_COMMAND", "aws")
    command = [aws_command, "s3", "sync"]
    local = Path(os.environ.get("DATA_DIR_PROCESSED", "data")) / args.subfolder
    remote = f"s3://robust-rearrangement/{args.subfolder}"

    if args.up:
        command += [str(local), remote]
    else:
        command += [remote, str(local)]

    if args.delete:
        command.append("--delete")

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
