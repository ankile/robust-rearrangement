import pickle
import tarfile


def zipped_img_generator(filename, max_samples=1000):
    n_samples = 0
    with tarfile.open(filename, "r:gz") as tar:
        for member in tar:
            if member.isfile() and ".pkl" in member.name:  # Replace 'your_condition' with actual condition
                with tar.extractfile(member) as f:
                    if f is not None:
                        content = f.read()
                        data = pickle.loads(content)
                        n_samples += 1

                        yield data

                        if n_samples >= max_samples:
                            break
