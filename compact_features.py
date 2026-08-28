"""Derives the compact 4-block feature vector from the (30, 447) arrays extraction.py writes.

Blocks, in this order, so each ablation arm is a prefix of the vector:

    A  hands            126   both hands, 21 points x 3, unchanged
    B  articulator pose  18   shoulders/elbows/wrists (pose 11..16) x 3, unchanged
    C  postural NMS       3   head roll, yaw, pitch
    D  facial NMS         4   mouth aperture, mouth width, eyebrow raise left/right
                        ---
                        151

Blocks A and B stay shoulder-normalized, exactly as stored. Blocks C and D are an
angle and distance ratios, which the shoulder normalization (translation + uniform
scale) leaves untouched, so they are read straight off the same array. No re-extraction.
"""

import numpy as np
import os

# Face-mesh ids kept by extraction.py. It filters enumerate(face_landmarks), so the
# stored order is ascending id, not the order they are listed in that script.
SELECTED_FACE_IDS = sorted([
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181,
    191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 415,
    46, 52, 53, 55, 65, 70, 105, 107, 276, 282, 283, 285, 295, 300, 334, 336,
    50, 118, 123, 137, 205, 206, 207, 212, 214, 216,
    280, 347, 352, 366, 425, 426, 427, 432, 434, 436,
])
SLOT = {mid: i for i, mid in enumerate(SELECTED_FACE_IDS)}

# Eyebrow groups, split by the side of the image they sit on (checked against the data:
# every id below is consistently on one side). Pose 2 is the left eye, pose 5 the right.
BROW_LEFT = [SLOT[i] for i in (276, 282, 283, 285, 295, 300, 334, 336)]
BROW_RIGHT = [SLOT[i] for i in (46, 52, 53, 55, 65, 70, 105, 107)]
LIP_TOP, LIP_BOTTOM = SLOT[13], SLOT[14]
LIP_CORNERS = SLOT[61], SLOT[291]

DIM_RAW = 447
DIM_TOTAL = 151
# Cut points for the four ablation arms: hands, +pose, +postural NMS, +facial NMS.
ARMS = {'A': 126, 'AB': 144, 'ABC': 147, 'ABCD': 151}

SRC_PATH = 'features'
DST_PATH = 'features_compact'


def derive_frame(frame):
    """One (447,) shoulder-normalized frame -> one (151,) compact frame."""
    pose = frame[0:99].reshape(33, 3)
    face = frame[99:321].reshape(74, 3)
    out = np.zeros(DIM_TOTAL)

    out[0:126] = frame[321:447]   # A: left hand then right hand
    out[126:144] = frame[33:51]   # B: pose 11..16

    # extraction.py stores zeros when a block was not detected. Ratios need real
    # landmarks, so leave C and D at zero rather than dividing by nothing.
    if not pose.any():
        return out
    inter_ocular = np.linalg.norm(pose[2, :2] - pose[5, :2])
    if inter_ocular < 1e-6:
        return out

    # C: head orientation, from the pose block alone.
    dx, dy = pose[2, :2] - pose[5, :2]                 # right eye -> left eye
    out[144] = np.arctan2(dy, dx)                      # roll
    ear_dist = np.linalg.norm(pose[7, :2] - pose[8, :2])
    if ear_dist > 1e-6:
        ear_mid_x = (pose[7, 0] + pose[8, 0]) / 2
        out[145] = (pose[0, 0] - ear_mid_x) / ear_dist  # yaw
    eye_mid_y = (pose[2, 1] + pose[5, 1]) / 2
    # ponytail: pitch off 2D y only. In projection a nod and a lowered head both just
    # move the nose down. If it reads as noise, try pose z, or drop pitch and shift the
    # block D indices down by one.
    out[146] = (pose[0, 1] - eye_mid_y) / inter_ocular  # pitch

    # D: facial NMS, every entry a distance over inter-ocular. y grows downward, so
    # eye_y - brow_y is positive when the brow is up.
    if not face.any():
        return out
    out[147] = abs(face[LIP_BOTTOM, 1] - face[LIP_TOP, 1]) / inter_ocular
    out[148] = np.linalg.norm(face[LIP_CORNERS[1], :2] - face[LIP_CORNERS[0], :2]) / inter_ocular
    out[149] = (pose[2, 1] - face[BROW_LEFT, 1].mean()) / inter_ocular
    out[150] = (pose[5, 1] - face[BROW_RIGHT, 1].mean()) / inter_ocular
    return out


def derive(seq):
    """(T, 447) -> (T, 151)."""
    return np.stack([derive_frame(f) for f in seq])


def mirror(compact):
    """Left-right mirror of a compact frame or (T, 151) sequence.

    Every block transforms obviously here: no landmark index remapping to get wrong.
    """
    m = np.asarray(compact, dtype=float)
    lead = m.shape[:-1]
    out = m.copy()

    hands = m[..., 0:126].reshape(lead + (2, 21, 3))[..., ::-1, :, :].copy()
    hands[..., 0] *= -1
    out[..., 0:126] = hands.reshape(lead + (126,))

    artic = m[..., 126:144].reshape(lead + (3, 2, 3))[..., ::-1, :].copy()
    artic[..., 0] *= -1
    out[..., 126:144] = artic.reshape(lead + (18,))

    out[..., 144] = -m[..., 144]   # roll
    out[..., 145] = -m[..., 145]   # yaw
    out[..., 149] = m[..., 150]    # eyebrow raise swaps sides
    out[..., 150] = m[..., 149]
    return out                     # pitch, mouth aperture and mouth width are unchanged


def derive_dir(src=SRC_PATH, dst=DST_PATH):
    """Convert every (T, 447) .npy under src into a (T, 151) .npy under dst."""
    written = 0
    for action in sorted(os.listdir(src)):
        in_dir = os.path.join(src, action)
        if not os.path.isdir(in_dir):
            continue
        out_dir = os.path.join(dst, action)
        os.makedirs(out_dir, exist_ok=True)
        for name in sorted(os.listdir(in_dir)):
            if not name.endswith('.npy'):
                continue
            out_path = os.path.join(out_dir, name)
            if os.path.exists(out_path):
                continue
            seq = np.load(os.path.join(in_dir, name))
            if seq.ndim != 2 or seq.shape[1] != DIM_RAW:
                print(f"Skipped {action}/{name}: expected (T, {DIM_RAW}), got {seq.shape}")
                continue
            np.save(out_path, derive(seq))
            written += 1
    print(f"Compact features: {written} new file(s) in {dst}/, {DIM_TOTAL} dims per frame")


def _self_check():
    rng = np.random.default_rng(0)
    roll = np.deg2rad(15.0)
    frame = np.zeros(DIM_RAW)
    pose = frame[0:99].reshape(33, 3)
    face = frame[99:321].reshape(74, 3)

    eye_mid = np.array([0.0, -1.2])
    pose[2, :2] = eye_mid + 0.1 * np.array([np.cos(roll), np.sin(roll)])   # left eye
    pose[5, :2] = eye_mid - 0.1 * np.array([np.cos(roll), np.sin(roll)])   # right eye
    pose[7, :2], pose[8, :2] = [0.18, -1.2], [-0.18, -1.2]                 # ears, 0.36 apart
    pose[0, :2] = [0.036, -1.15]                                           # nose
    pose[11:17] = rng.normal(size=(6, 3))                                  # articulators
    face[LIP_TOP, 1], face[LIP_BOTTOM, 1] = -0.90, -0.84
    face[LIP_CORNERS[0], :2], face[LIP_CORNERS[1], :2] = [-0.08, -0.87], [0.08, -0.87]
    face[BROW_LEFT, 1], face[BROW_RIGHT, 1] = -1.35, -1.30
    frame[321:447] = rng.normal(size=126)

    c = derive_frame(frame)
    assert c.shape == (DIM_TOTAL,)
    assert np.allclose(c[0:126], frame[321:447])
    assert np.allclose(c[126:144], frame[33:51])
    assert np.isclose(c[144], roll), c[144]
    assert np.isclose(c[145], 0.036 / 0.36)
    assert np.isclose(c[146], 0.05 / 0.2)
    assert np.isclose(c[147], 0.06 / 0.2)
    assert np.isclose(c[148], 0.16 / 0.2)
    assert np.isclose(c[149], (pose[2, 1] + 1.35) / 0.2)
    assert np.isclose(c[150], (pose[5, 1] + 1.30) / 0.2)

    # The claim blocks C and D rest on: they survive any translation + uniform scale,
    # so re-running the shoulder normalization cannot move them.
    moved = derive_frame(frame * 2.5 + 0.3)
    assert np.allclose(moved[144:151], c[144:151])

    m = mirror(c)
    assert np.allclose(mirror(m), c)                       # mirroring twice is identity
    assert np.isclose(m[144], -c[144]) and np.isclose(m[145], -c[145])
    assert np.isclose(m[146], c[146])                      # pitch survives
    assert np.allclose(m[147:149], c[147:149])             # mouth survives
    assert np.isclose(m[149], c[150]) and np.isclose(m[150], c[149])
    flipped_rh = frame[384:447].reshape(21, 3) * [-1, 1, 1]
    assert np.allclose(m[0:63], flipped_rh.ravel())        # right hand lands in the left slot
    assert np.allclose(m[126:129], frame[36:39] * [-1, 1, 1])  # shoulder 12 lands in 11's slot

    # Missing detections must not leak a division.
    assert not derive_frame(np.zeros(DIM_RAW)).any()
    no_face = frame.copy()
    no_face[99:321] = 0
    nf = derive_frame(no_face)
    assert nf[144:147].any() and not nf[147:151].any()

    assert derive(np.stack([frame, frame])).shape == (2, DIM_TOTAL)
    assert np.allclose(mirror(np.stack([c, c])), np.stack([m, m]))
    print("compact_features self-check passed")


if __name__ == '__main__':
    _self_check()
    if os.path.isdir(SRC_PATH):
        derive_dir()
