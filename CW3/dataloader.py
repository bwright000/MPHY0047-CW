# Data loader for MPHY0047 Coursework 3.
# Loads JIGSAWS kinematic data for 8 participants × 5 trials × 3 tasks.
# Each trial is an (N_samples, 76) array of kinematic variables.
# Also loads COVID-19 CT images (349 COVID + 397 Non-COVID).

import os
import zipfile
import numpy as np
import scipy.io as sio
from skimage.io import imread
from skimage.transform import resize

# Resolve paths relative to this file
_DIR = os.path.dirname(os.path.abspath(__file__))
_PROVIDED = os.path.join(_DIR, 'Provided', 'Provided')

# Load JIGSAWS data
cw3_data = sio.loadmat(os.path.join(_PROVIDED, 'cw3_a', 'jigsaw_matlab.mat'), simplify_cells=False)
suturing = cw3_data['suturing']             # (8, 5) object array
needle_passing = cw3_data['needle_passing']
knot_tying = cw3_data['knot_tying']

# GLOBAL VARIABLES
PARTICIPANTS = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
EXPERTS = ['C', 'D', 'E', 'F']
NOVICES = ['B', 'G', 'H', 'I']
EXPERT_INDICES = [1, 2, 3, 4]
NOVICE_INDICES = [0, 5, 6, 7]
NUM_PARTICIPANTS = 8
NUM_TRIALS = 5

# Sampling frequency (Hz)
FS = 30
DT = 1.0 / FS

# Column indices (0-based) for the 76 kinematic variables.
# Each manipulator block: xyz (3) | R flattened (9) | trans_vel (3) | rot_vel (3) | gripper (1)

# Master left: cols 0-18
ML_XYZ  = slice(0, 3)
ML_ROT  = slice(3, 12)
ML_TVEL = slice(12, 15)
ML_RVEL = slice(15, 18)
ML_GRIP = 18

# Master right: cols 19-37
MR_XYZ  = slice(19, 22)
MR_ROT  = slice(22, 31)
MR_TVEL = slice(31, 34)
MR_RVEL = slice(34, 37)
MR_GRIP = 37

# Slave left: cols 38-56
SL_XYZ  = slice(38, 41)
SL_ROT  = slice(41, 50)
SL_TVEL = slice(50, 53)
SL_RVEL = slice(53, 56)
SL_GRIP = 56

# Slave right: cols 57-75
SR_XYZ  = slice(57, 60)
SR_ROT  = slice(60, 69)
SR_TVEL = slice(69, 72)
SR_RVEL = slice(72, 75)
SR_GRIP = 75

# COVID CT paths
_CW3B = os.path.join(_PROVIDED, 'cw3_b')
COVID_ZIP     = os.path.join(_CW3B, 'CT_COVID.zip')
NON_COVID_ZIP = os.path.join(_CW3B, 'CT_NonCOVID.zip')
COVID_DIR     = os.path.join(_CW3B, 'CT_COVID')
NON_COVID_DIR = os.path.join(_CW3B, 'CT_NonCOVID')
IMAGE_SIZE = (120, 120)


def get_label(participant_idx):
    """Return 1 for expert, 0 for novice."""
    return 1 if participant_idx in EXPERT_INDICES else 0


def load_covid_images(directory, label, target_size=IMAGE_SIZE):
    """
    Load all jpg/png images from a directory, resize to target_size.
    Returns lists of (images, labels, filenames).
    """
    images, labels, filenames = [], [], []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(('.jpg', '.png')):
            img = imread(os.path.join(directory, fname), as_gray=True)
            img = resize(img, target_size, anti_aliasing=True)
            images.append(img)
            labels.append(label)
            filenames.append(fname)
    return images, labels, filenames


def load_covid_data(target_size=IMAGE_SIZE):
    """
    Load and resize COVID and Non-COVID CT images.
    Returns (X, y, filenames) where X is (N, h, w), y is (N,).
    """
    # Extract zips if needed
    for zpath, ddir in [(COVID_ZIP, COVID_DIR), (NON_COVID_ZIP, NON_COVID_DIR)]:
        if not os.path.isdir(ddir):
            with zipfile.ZipFile(zpath, 'r') as zf:
                zf.extractall(os.path.dirname(ddir))

    c_imgs, c_labels, c_fnames = load_covid_images(COVID_DIR, 1, target_size)
    nc_imgs, nc_labels, nc_fnames = load_covid_images(NON_COVID_DIR, 0, target_size)

    X = np.array(c_imgs + nc_imgs)
    y = np.array(c_labels + nc_labels)
    filenames = c_fnames + nc_fnames
    return X, y, filenames


def load_covid_data_imbalanced(target_size=IMAGE_SIZE):
    """
    Load COVID CT data with imbalanced classes (Q5).
    Removes the last 200 Non-COVID images → 349 COVID vs ~197 Non-COVID.
    """
    for zpath, ddir in [(COVID_ZIP, COVID_DIR), (NON_COVID_ZIP, NON_COVID_DIR)]:
        if not os.path.isdir(ddir):
            with zipfile.ZipFile(zpath, 'r') as zf:
                zf.extractall(os.path.dirname(ddir))

    c_imgs, c_labels, c_fnames = load_covid_images(COVID_DIR, 1, target_size)
    nc_imgs, nc_labels, nc_fnames = load_covid_images(NON_COVID_DIR, 0, target_size)

    # Remove last 200 non-COVID images
    nc_imgs = nc_imgs[:-200]
    nc_labels = nc_labels[:-200]
    nc_fnames = nc_fnames[:-200]

    X = np.array(c_imgs + nc_imgs)
    y = np.array(c_labels + nc_labels)
    filenames = c_fnames + nc_fnames
    return X, y, filenames


def tests():
    print("JIGSAWS loaded successfully.")
    print(f"Suturing shape: {suturing.shape}")
    for p in range(NUM_PARTICIPANTS):
        shapes = [suturing[p, t].shape for t in range(NUM_TRIALS)]
        label = "expert" if p in EXPERT_INDICES else "novice"
        print(f"  {PARTICIPANTS[p]} ({label}): {shapes}")

# tests()
