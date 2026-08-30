"""Derives the compact 4-block feature vector from the (30, 447) arrays extraction.py writes.

Blocks, in this order, so each ablation arm is a prefix of the vector:

    A  hands            126   both hands, 21 points x 3, unchanged
    B  articulator pose  13   upper-arm and forearm unit vectors x 2 sides, + shoulder tilt
    C  postural NMS       3   head roll, yaw, pitch
    D  facial NMS         4   mouth aperture, mouth width, eyebrow raise left/right
                        ---
                        146

Block A stays shoulder-normalized coordinates, exactly as stored. Blocks B, C and D
are directions, angles and distance ratios, all invariant to translation and uniform
scale, so the shoulder normalization cannot move them and they are read straight off
the same array. No re-extraction.

Block B was 18 raw coordinates (pose 11..16). Measured on the four-signer set, adding
those 18 numbers to the hands raised linear signer-recoverability from 65.5% to 94.6%,
+29pp, because elbow and wrist coordinates encode arm length and reach. Per joint, the
between-signer spread over the within-signer spread came out at 0.62 for the shoulders
but 1.30-1.51 for elbows and wrists: the shoulders carry no identity, the limbs do.

Unit vectors keep every bit of arm motion and drop only limb length. Nothing is lost
by dropping wrist position, because hand landmark 0 is the wrist and already sits in
block A. This is Next-Steps 4.2 ("absolute geometry -> relative deformation") applied
to block B, which the document applies only to C and D.
"""

import numpy as np
import os
import sys
import tempfile

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

# extraction.py tests membership with `i not in SELECTED_FACE_IDS`, so a duplicated id
# is emitted once but counted twice here, which shifts every face slot after it.
assert len(SELECTED_FACE_IDS) == len(set(SELECTED_FACE_IDS)), "duplicate face id"

# Eyebrow groups, split by the side of the image they sit on (checked against the data:
# every id below is consistently on one side). Pose 2 is the left eye, pose 5 the right.
BROW_LEFT = [SLOT[i] for i in (276, 282, 283, 285, 295, 300, 334, 336)]
BROW_RIGHT = [SLOT[i] for i in (46, 52, 53, 55, 65, 70, 105, 107)]
LIP_TOP, LIP_BOTTOM = SLOT[13], SLOT[14]
LIP_CORNERS = SLOT[61], SLOT[291]

# Layout of the array extraction.py writes: pose, face, left hand, right hand. Derived
# from the id list above, so changing the face selection moves every boundary at once
# instead of leaving literals here to update by hand.
N_FACE = len(SELECTED_FACE_IDS)
POSE_END = 33 * 3
FACE_END = POSE_END + N_FACE * 3
DIM_RAW = FACE_END + 2 * 21 * 3   # 447 for the current 74-point face selection

DIM_TOTAL = 146
# Cut points for the four ablation arms: hands, +pose, +postural NMS, +facial NMS.
ARMS = {'A': 126, 'AB': 139, 'ABC': 142, 'ABCD': 146}

# MediaPipe Pose: 11/12 shoulders, 13/14 elbows, 15/16 wrists. Left first in each pair,
# so block B reshapes to (2 sides, 2 segments, 3) and mirroring is an axis reversal.
LIMBS = ((11, 13), (13, 15), (12, 14), (14, 16))   # L upper, L fore, R upper, R fore


def _unit(v):
    """Direction of v, or zeros when the joint was not detected."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros(3)

SRC_PATH = 'features'
DST_PATH = 'features_compact'


def derive_frame(frame):
    """One shoulder-normalized (DIM_RAW,) frame -> one compact (DIM_TOTAL,) frame."""
    pose = frame[0:POSE_END].reshape(33, 3)
    face = frame[POSE_END:FACE_END].reshape(N_FACE, 3)
    out = np.zeros(DIM_TOTAL)

    out[0:126] = frame[FACE_END:DIM_RAW]      # A: left hand then right hand

    # extraction.py stores zeros when a block was not detected. Directions and ratios
    # need real landmarks, so leave B, C and D at zero rather than dividing by nothing.
    if not pose.any():
        return out

    # B: articulator DIRECTION, not position. Each limb contributes its unit vector,
    # so the arm's configuration survives and its length does not. No denominator is
    # needed or wanted: a unit vector is already scale-free, and dividing by an
    # inter-ocular distance would mix a face measurement back into an arm measurement.
    for k, (a, b) in enumerate(LIMBS):
        out[126 + 3 * k:129 + 3 * k] = _unit(pose[b] - pose[a])
    # Shoulder tilt. Shoulder normalization centres on the shoulder midpoint and scales
    # by the shoulder distance without rotating, so after it the two shoulders sit at
    # +/-0.5 of the shoulder unit vector and their coordinates encode the tilt and
    # nothing else. Kept as one scalar so Next-Steps 4.8 can be measured, not assumed.
    out[138] = np.arctan2(*(pose[11, :2] - pose[12, :2])[::-1])

    inter_ocular = np.linalg.norm(pose[2, :2] - pose[5, :2])
    if inter_ocular < 1e-6:
        return out

    # C: head orientation, from the pose block alone.
    dx, dy = pose[2, :2] - pose[5, :2]                 # right eye -> left eye
    out[139] = np.arctan2(dy, dx)                      # roll
    ear_dist = np.linalg.norm(pose[7, :2] - pose[8, :2])
    if ear_dist > 1e-6:
        ear_mid_x = (pose[7, 0] + pose[8, 0]) / 2
        out[140] = (pose[0, 0] - ear_mid_x) / ear_dist  # yaw
    eye_mid_y = (pose[2, 1] + pose[5, 1]) / 2
    # ponytail: pitch off 2D y only. In projection a nod and a lowered head both just
    # move the nose down. If it reads as noise, try pose z, or drop pitch and shift the
    # block D indices down by one.
    out[141] = (pose[0, 1] - eye_mid_y) / inter_ocular  # pitch

    # D: facial NMS, every entry a distance over inter-ocular. y grows downward, so
    # eye_y - brow_y is positive when the brow is up.
    if not face.any():
        return out
    out[142] = abs(face[LIP_BOTTOM, 1] - face[LIP_TOP, 1]) / inter_ocular
    out[143] = np.linalg.norm(face[LIP_CORNERS[1], :2] - face[LIP_CORNERS[0], :2]) / inter_ocular
    out[144] = (pose[2, 1] - face[BROW_LEFT, 1].mean()) / inter_ocular
    out[145] = (pose[5, 1] - face[BROW_RIGHT, 1].mean()) / inter_ocular
    return out


def derive(seq):
    """(T, DIM_RAW) -> (T, DIM_TOTAL)."""
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

    # (2 sides, 2 segments, 3): reversing the side axis swaps the arms, and negating x
    # flips each direction. Limb order in LIMBS is left-then-right for this reason.
    artic = m[..., 126:138].reshape(lead + (2, 2, 3))[..., ::-1, :, :].copy()
    artic[..., 0] *= -1
    out[..., 126:138] = artic.reshape(lead + (12,))

    out[..., 138] = -m[..., 138]   # shoulder tilt
    out[..., 139] = -m[..., 139]   # roll
    out[..., 140] = -m[..., 140]   # yaw
    out[..., 144] = m[..., 145]    # eyebrow raise swaps sides
    out[..., 145] = m[..., 144]
    return out                     # pitch, mouth aperture and mouth width are unchanged


def derive_dir(src=SRC_PATH, dst=DST_PATH, force=False):
    """Convert every (T, DIM_RAW) .npy under src into a (T, DIM_TOTAL) .npy under dst.

    Output that already exists is left alone unless force is set. Pass force=True after
    changing any block definition, or the stale vectors survive and the run looks like a
    no-op.
    """
    written, kept = 0, 0
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
            if os.path.exists(out_path) and not force:
                kept += 1
                continue
            seq = np.load(os.path.join(in_dir, name))
            if seq.ndim != 2 or seq.shape[1] != DIM_RAW:
                print(f"Skipped {action}/{name}: expected (T, {DIM_RAW}), got {seq.shape}")
                continue
            np.save(out_path, derive(seq))
            written += 1
    print(f"Compact features -> {dst}/: {written} written, {kept} kept, "
          f"{DIM_TOTAL} dims per frame")
    if kept:
        print("  Kept files were NOT re-derived. Rerun with --force after changing a block.")


def _self_check():
    rng = np.random.default_rng(0)
    roll = np.deg2rad(15.0)
    frame = np.zeros(DIM_RAW)
    pose = frame[0:POSE_END].reshape(33, 3)
    face = frame[POSE_END:FACE_END].reshape(N_FACE, 3)

    eye_mid = np.array([0.0, -1.2])
    pose[2, :2] = eye_mid + 0.1 * np.array([np.cos(roll), np.sin(roll)])   # left eye
    pose[5, :2] = eye_mid - 0.1 * np.array([np.cos(roll), np.sin(roll)])   # right eye
    pose[7, :2], pose[8, :2] = [0.18, -1.2], [-0.18, -1.2]                 # ears, 0.36 apart
    pose[0, :2] = [0.036, -1.15]                                           # nose
    pose[11:17] = rng.normal(size=(6, 3))                                  # articulators
    face[LIP_TOP, 1], face[LIP_BOTTOM, 1] = -0.90, -0.84
    face[LIP_CORNERS[0], :2], face[LIP_CORNERS[1], :2] = [-0.08, -0.87], [0.08, -0.87]
    face[BROW_LEFT, 1], face[BROW_RIGHT, 1] = -1.35, -1.30
    frame[FACE_END:DIM_RAW] = rng.normal(size=126)

    c = derive_frame(frame)
    assert c.shape == (DIM_TOTAL,)
    assert np.allclose(c[0:126], frame[FACE_END:DIM_RAW])
    for k, (a, b) in enumerate(LIMBS):                     # B: direction, unit length
        seg = c[126 + 3 * k:129 + 3 * k]
        assert np.isclose(np.linalg.norm(seg), 1.0)
        assert np.allclose(seg, _unit(pose[b] - pose[a]))
    assert np.isclose(c[138], np.arctan2(*(pose[11, :2] - pose[12, :2])[::-1]))
    assert np.isclose(c[139], roll), c[139]
    assert np.isclose(c[140], 0.036 / 0.36)
    assert np.isclose(c[141], 0.05 / 0.2)
    assert np.isclose(c[142], 0.06 / 0.2)
    assert np.isclose(c[143], 0.16 / 0.2)
    assert np.isclose(c[144], (pose[2, 1] + 1.35) / 0.2)
    assert np.isclose(c[145], (pose[5, 1] + 1.30) / 0.2)

    # The claim blocks B, C and D rest on: they survive any translation + uniform
    # scale, so re-running the shoulder normalization cannot move them.
    moved = derive_frame(frame * 2.5 + 0.3)
    assert np.allclose(moved[126:DIM_TOTAL], c[126:DIM_TOTAL])

    # The point of the change: stretching one arm must not move block B. Lengthening
    # the forearm along its own direction changes reach, which is the anthropometry
    # the old coordinate form leaked, and leaves every direction untouched.
    longer = frame.copy()
    lp = longer[0:POSE_END].reshape(33, 3)
    lp[15] = lp[13] + 1.7 * (lp[15] - lp[13])              # left forearm, 70% longer
    lc = derive_frame(longer)
    assert np.allclose(lc[126:DIM_TOTAL], c[126:DIM_TOTAL]), "block B still encodes limb length"

    # ...while a genuine change of arm configuration must still show up.
    bent = frame.copy()
    bp = bent[0:POSE_END].reshape(33, 3)
    bp[15] = bp[13] + np.array([0.4, -0.3, 0.1])
    assert not np.allclose(derive_frame(bent)[129:132], c[129:132]), "block B lost arm motion"

    m = mirror(c)
    assert np.allclose(mirror(m), c)                       # mirroring twice is identity
    assert np.isclose(m[138], -c[138])                     # shoulder tilt
    assert np.isclose(m[139], -c[139]) and np.isclose(m[140], -c[140])
    assert np.isclose(m[141], c[141])                      # pitch survives
    assert np.allclose(m[142:144], c[142:144])             # mouth survives
    assert np.isclose(m[144], c[145]) and np.isclose(m[145], c[144])
    flipped_rh = frame[FACE_END + 63:DIM_RAW].reshape(21, 3) * [-1, 1, 1]
    assert np.allclose(m[0:63], flipped_rh.ravel())        # right hand lands in the left slot
    # right upper arm lands in the left upper arm's slot, x negated
    assert np.allclose(m[126:129], c[132:135] * [-1, 1, 1])
    assert np.allclose(m[129:132], c[135:138] * [-1, 1, 1])

    # Missing detections must not leak a division.
    assert not derive_frame(np.zeros(DIM_RAW)).any()
    no_face = frame.copy()
    no_face[POSE_END:FACE_END] = 0
    nf = derive_frame(no_face)
    assert nf[126:142].any() and not nf[142:DIM_TOTAL].any()

    assert derive(np.stack([frame, frame])).shape == (2, DIM_TOTAL)
    assert np.allclose(mirror(np.stack([c, c])), np.stack([m, m]))

    # force must really overwrite: a stale block C silently poisons a whole ablation arm.
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, 'in'), os.path.join(tmp, 'out')
        os.makedirs(os.path.join(src, 'x'))
        np.save(os.path.join(src, 'x', 'a.npy'), np.stack([frame, frame]))
        derive_dir(src, dst)
        stale = os.path.join(dst, 'x', 'a.npy')
        np.save(stale, np.zeros((2, DIM_TOTAL)))
        derive_dir(src, dst)
        assert not np.load(stale).any(), "existing output should be kept by default"
        derive_dir(src, dst, force=True)
        assert np.load(stale).any(), "force should overwrite existing output"

    print("compact_features self-check passed")


if __name__ == '__main__':
    _self_check()
    if os.path.isdir(SRC_PATH):
        derive_dir(force='--force' in sys.argv)
