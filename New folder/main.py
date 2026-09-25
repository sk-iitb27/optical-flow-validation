import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter


# ============================================================
# USER SETTINGS
# ============================================================

VIDEO_PATH = "optical_flow_vdo.mp4"

# Paper used wm = 5 and wt = 3 for the cantilever experiment
WM = 5.0
WT = 3.0

# Choose the orientation of your structural edge
#
# "vertical"   -> beam/column edge is approximately vertical
#                 -> primarily measures horizontal displacement
#
# "horizontal" -> edge is approximately horizontal
#                 -> primarily measures vertical displacement
#
EDGE_DIRECTION = "vertical"

# Which row/column to use for the single-point validation plot
# For vertical edge: choose a row
# For horizontal edge: choose a column
REFERENCE_INDEX = None

# Pixel-to-mm conversion.
# If your experiment has calibration:
# e.g. 100 pixels = 50 mm -> PIXEL_TO_MM = 0.5
PIXEL_TO_MM = 1.0

# Save output
OUTPUT_CSV = "bhowmick_displacement.csv"


# ============================================================
# STEP 1: READ VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\nVideo information")
print("-----------------")
print("FPS       :", fps)
print("Frames    :", n_frames)

frames = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frames.append(gray.astype(np.float64))

cap.release()

frames = np.asarray(frames)

print("Video loaded:", frames.shape)


# ============================================================
# STEP 2: SELECT ROI
# ============================================================

# Use first frame for ROI selection

first_frame = frames[0].astype(np.uint8)

print("\nSelect the bounding box around the vibrating edge.")
print("Press ENTER after selecting the ROI.")

x, y, w, h = cv2.selectROI(
    "Select structural edge",
    first_frame,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

if w == 0 or h == 0:
    raise RuntimeError("No ROI selected.")

frames = frames[:, y:y+h, x:x+w]

print("\nROI size:", frames.shape)


# ============================================================
# STEP 3: NORMALIZE VIDEO
# ============================================================

frames = frames / 255.0


# ============================================================
# STEP 4: PARAMETERS OF 3-D DALEMBERTIAN OF GAUSSIAN
# ============================================================

# Paper:
#
# wm = 2 * sqrt(2) * sigma
#
# wt = wm / u
#
# Therefore:
#
# sigma = wm / (2*sqrt(2))
# u = wm / wt
#
# temporal sigma = sigma/u

sigma_space = WM / (2.0 * np.sqrt(2.0))

u = WM / WT

sigma_time = sigma_space / u

print("\nFilter parameters")
print("-----------------")
print("Spatial sigma :", sigma_space)
print("Velocity scale u :", u)
print("Temporal sigma :", sigma_time)


# ============================================================
# STEP 5: 3-D GAUSSIAN DERIVATIVES
# ============================================================

# Array ordering:
#
# frames[time, y, x]
#
# Therefore sigma:
#
# (time, y, x)

sigma = (
    sigma_time,
    sigma_space,
    sigma_space
)

print("\nComputing 3-D Gaussian derivatives...")
print("This may take some time for long videos.")


# Smoothed video

G = gaussian_filter(
    frames,
    sigma=sigma,
    order=(0, 0, 0),
    mode="reflect"
)


# Second derivative with respect to x

G_xx = gaussian_filter(
    frames,
    sigma=sigma,
    order=(0, 0, 2),
    mode="reflect"
)


# Second derivative with respect to y

G_yy = gaussian_filter(
    frames,
    sigma=sigma,
    order=(0, 2, 0),
    mode="reflect"
)


# Second derivative with respect to time

G_tt = gaussian_filter(
    frames,
    sigma=sigma,
    order=(2, 0, 0),
    mode="reflect"
)


# ============================================================
# STEP 6: DALEMBERTIAN OF GAUSSIAN
# ============================================================

# Paper uses the positive form:
#
# -(d²/dx² + d²/dy²) + (1/u²)d²/dt²
#
# applied to the Gaussian-filtered video.
#
# Therefore:
#
# S = -(G_xx + G_yy) + G_tt/u²

S = -(G_xx + G_yy) + G_tt / (u ** 2)

print("Edge signal calculated.")


# ============================================================
# STEP 7: SUB-PIXEL ZERO CROSSING
# ============================================================

def subpixel_zero_crossing(values):
    """
    Estimate zero crossing using local linear interpolation.

    values = [S(-1), S(0), S(+1)]
    """

    left = values[0]
    center = values[1]
    right = values[2]

    # crossing between -1 and 0
    if left * center <= 0 and left != center:

        offset = -left / (center - left)

        return -1 + offset

    # crossing between 0 and +1
    if center * right <= 0 and center != right:

        offset = -center / (right - center)

        return offset

    return np.nan


# ============================================================
# STEP 8: FIND EDGE POSITIONS
# ============================================================

T, H, W = S.shape

print("\nTracking edge...")

if EDGE_DIRECTION.lower() == "vertical":

    # For a vertical structural edge:
    #
    # Each row is searched horizontally.
    #
    # x-position of edge is obtained.

    displacement = np.full((H, T), np.nan)

    for t in range(T):

        for row in range(H):

            signal = S[t, row, :]

            crossings = []

            for col in range(1, W - 1):

                if signal[col - 1] * signal[col + 1] <= 0:

                    values = [
                        signal[col - 1],
                        signal[col],
                        signal[col + 1]
                    ]

                    offset = subpixel_zero_crossing(values)

                    if not np.isnan(offset):

                        position = col + offset

                        crossings.append(position)

            # Usually there should be one dominant structural edge.
            if len(crossings) > 0:

                # Select strongest crossing based on local gradient
                best = None
                best_gradient = -np.inf

                for pos in crossings:

                    c = int(round(pos))

                    if 1 <= c < W - 1:

                        gradient = abs(
                            signal[c + 1] - signal[c - 1]
                        )

                        if gradient > best_gradient:

                            best_gradient = gradient
                            best = pos

                displacement[row, t] = best


elif EDGE_DIRECTION.lower() == "horizontal":

    # For horizontal structural edge:
    #
    # Each column is searched vertically.

    displacement = np.full((W, T), np.nan)

    for t in range(T):

        for col in range(W):

            signal = S[t, :, :][:, col]

            crossings = []

            for row in range(1, H - 1):

                if signal[row - 1] * signal[row + 1] <= 0:

                    values = [
                        signal[row - 1],
                        signal[row],
                        signal[row + 1]
                    ]

                    offset = subpixel_zero_crossing(values)

                    if not np.isnan(offset):

                        position = row + offset

                        crossings.append(position)

            if len(crossings) > 0:

                best = None
                best_gradient = -np.inf

                for pos in crossings:

                    r = int(round(pos))

                    if 1 <= r < H - 1:

                        gradient = abs(
                            signal[r + 1] - signal[r - 1]
                        )

                        if gradient > best_gradient:

                            best_gradient = gradient
                            best = pos

                displacement[col, t] = best

else:

    raise ValueError(
        "EDGE_DIRECTION must be 'vertical' or 'horizontal'"
    )


# ============================================================
# STEP 9: REMOVE BAD / MISSING VALUES
# ============================================================

print("Cleaning displacement data...")

for i in range(displacement.shape[0]):

    signal = displacement[i, :]

    good = np.isfinite(signal)

    if np.sum(good) > 2:

        displacement[i, :] = np.interp(
            np.arange(T),
            np.where(good)[0],
            signal[good]
        )


# ============================================================
# STEP 10: CONVERT EDGE POSITION TO DISPLACEMENT
# ============================================================

# Lagrangian displacement:
#
# displacement(t) =
# position(t) - initial position

for i in range(displacement.shape[0]):

    initial_position = displacement[i, 0]

    displacement[i, :] -= initial_position


# Convert pixels to physical units

displacement_physical = displacement * PIXEL_TO_MM


# ============================================================
# STEP 11: TIME VECTOR
# ============================================================

time = np.arange(T) / fps


# ============================================================
# STEP 12: SAVE FULL-FIELD DISPLACEMENT
# ============================================================

if EDGE_DIRECTION.lower() == "vertical":

    data = {
        "time_s": time
    }

    for i in range(H):

        data[f"row_{i}_disp_mm"] = displacement_physical[i, :]

else:

    data = {
        "time_s": time
    }

    for i in range(W):

        data[f"column_{i}_disp_mm"] = displacement_physical[i, :]


df = pd.DataFrame(data)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nSaved:")
print(OUTPUT_CSV)


# ============================================================
# STEP 13: SELECT REFERENCE LOCATION
# ============================================================

if REFERENCE_INDEX is None:

    REFERENCE_INDEX = displacement_physical.shape[0] // 2

reference_signal = displacement_physical[
    REFERENCE_INDEX,
    :
]


# ============================================================
# STEP 14: PLOT SINGLE-POINT DISPLACEMENT
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    reference_signal,
    linewidth=1.2
)

plt.xlabel("Time (s)")
plt.ylabel("Displacement (mm)")

plt.title(
    "Bhowmick et al. Edge-Based Full-Field Displacement"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# STEP 15: FULL-FIELD DISPLACEMENT MAP
# ============================================================

plt.figure(figsize=(10, 6))

plt.imshow(
    displacement_physical,
    aspect="auto",
    extent=[
        time[0],
        time[-1],
        0,
        displacement_physical.shape[0]
    ],
    origin="lower"
)

plt.colorbar(
    label="Displacement (mm)"
)

plt.xlabel("Time (s)")

if EDGE_DIRECTION.lower() == "vertical":
    plt.ylabel("Position along edge (pixel)")
else:
    plt.ylabel("Position along edge (pixel)")

plt.title(
    "Full-Field Displacement Time History"
)

plt.tight_layout()

plt.show()


print("\nFinished.")