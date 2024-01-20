import pickle
import tarfile
import cv2


def zipped_img_generator(filename, max_samples=1000):
    n_samples = 0
    with tarfile.open(filename, "r:gz") as tar:
        for member in tar:
            if (
                member.isfile() and ".pkl" in member.name
            ):  # Replace 'your_condition' with actual condition
                with tar.extractfile(member) as f:
                    if f is not None:
                        content = f.read()
                        data = pickle.loads(content)
                        n_samples += 1

                        yield data

                        if n_samples >= max_samples:
                            break


def resize(img):
    """Resizes `img` into 320x240."""
    th, tw = 240, 320
    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    return img


def resize_crop(img):
    """Resizes `img` and center crops into 320x240."""
    th, tw = 240, 320
    ch, cw = img.shape[:2]

    # Calculate the aspect ratio of the original image.
    aspect_ratio = cw / ch

    # Resize based on the width, keeping the aspect ratio constant.
    new_width = int(th * aspect_ratio)
    img = cv2.resize(img, (new_width, th), interpolation=cv2.INTER_AREA)

    # Calculate the crop size.
    crop_size = (new_width - tw) // 2

    # Crop the image.
    img = img[:, crop_size : new_width - crop_size]

    return img
