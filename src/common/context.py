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


@contextlib.contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


if __name__ == "__main__":
    # Example usage
    with suppress_print():
        print("This won't be printed.")
    print("This will be printed.")
