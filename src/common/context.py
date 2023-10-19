import sys
import contextlib

import os


@contextlib.contextmanager
def suppress_print(suppress=True):
    old_stdout = sys.stdout
    if suppress:
        sys.stdout = open("/dev/null", "w")
    try:
        yield
    finally:
        if suppress:
            sys.stdout.close()
        sys.stdout = old_stdout


@contextlib.contextmanager
def suppress_all_output(disable=True):
    if disable:
        null_fd = os.open(os.devnull, os.O_RDWR)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
    try:
        yield
    finally:
        if disable:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(null_fd)


if __name__ == "__main__":
    # Example usage
    with suppress_print():
        print("This won't be printed.")
    print("This will be printed.")
