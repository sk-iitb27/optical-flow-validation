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
#################################################33
#we are selecting the direction of the edge that we want to analyse, this direction will be used in th elate rpart of the code 



###################################################
# Which row/column to use for the single-point validation plot
# For vertical edge: choose a row
# For horizontal edge: choose a column
REFERENCE_INDEX = None

# Pixel-to-mm conversion.
# If your experiment has calibration:
# e.g. 100 pixels = 50 mm -> PIXEL_TO_MM = 0.5
PIXEL_TO_MM = 1.0

# Save output
OUTPUT_CSV = "displacement.csv"


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

    ret, frame = cap.read()   #ret tells us did we get a frame , like if it is at the end we will not get the frame 

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

frames = frames[:, y:y+h, x:x+w]  #frames (N,H,W) where N is the frame 

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

# Paper:
#
# G(x,y,t) ∝ exp[-alpha(x² + y² + u²t²)]
#
# The 3-D Gaussian is separable:
#
# G(x,y,t) = g(x) g(y) g(ut)
#
# where g is the 1-D Gaussian.
#
# The paper defines:
#
# wm = 2 * sqrt(2) * sigma
#
# wt = wm / u
#
# Therefore:
#
# sigma_space = wm / (2*sqrt(2))
# sigma_time  = sigma_space / u


print("\nComputing 3-D Gaussian derivatives...")
print("This may take some time for long videos.")


# ------------------------------------------------------------
# 1-D Gaussian
# ------------------------------------------------------------

# Spatial Gaussian:
#
# g(x) = 1/(sigma*sqrt(2*pi))
#        * exp(-x²/(2*sigma²))


spatial_radius = int(np.ceil(3.0 * sigma_space))  #np.ceil rounds upward to the next integer (numpy)

x = np.arange(
    -spatial_radius,
    spatial_radius + 1,
    dtype=np.float64    
)#(- 3*sigma , -2*sigma, -sigma, 0 , sigma, 2*sigma, 3*sigma )

g_space = (
    1.0 / (sigma_space * np.sqrt(2.0 * np.pi))
    * np.exp(
        -x**2 / (2.0 * sigma_space**2) #sigma_space = WM / (2.0 * np.sqrt(2.0))
    )
)


# ------------------------------------------------------------
# Second derivative of spatial Gaussian
# ------------------------------------------------------------

# Paper Appendix Eq. (28):
#
# g''(x) =
#
# 1/(sigma³*sqrt(2*pi))
# * (x²/sigma² - 1)
# * exp(-x²/(2*sigma²))


g_space_xx = (
    1.0
    / (
        sigma_space**3
        * np.sqrt(2.0 * np.pi)
    )
    * (
        x**2 / sigma_space**2 - 1.0
    )
    * np.exp(
        -x**2
        / (2.0 * sigma_space**2)
    )
)


# ------------------------------------------------------------
# Temporal Gaussian
# ------------------------------------------------------------

temporal_radius = int(np.ceil(3.0 * sigma_time))#repeating the same steps for the x  in time and y odmains 

t = np.arange(
    -temporal_radius,
    temporal_radius + 1,
    dtype=np.float64
)


# g(ut)

g_time = (
    1.0 / (sigma_time * np.sqrt(2.0 * np.pi))
    * np.exp(
        -t**2 / (2.0 * sigma_time**2)
    )
)


# ------------------------------------------------------------
# Second derivative of temporal Gaussian
# ------------------------------------------------------------

g_time_tt = (
    1.0
    / (
        sigma_time**3
        * np.sqrt(2.0 * np.pi)
    )
    * (
        t**2 / sigma_time**2 - 1.0
    )
    * np.exp(
        -t**2
        / (2.0 * sigma_time**2)
    )
)


# ------------------------------------------------------------
# Apply the separable Gaussian filters
# ------------------------------------------------------------

from scipy.ndimage import convolve1d
from scipy.ndimage import convolve

# # Gaussian smoothing in x

# G = convolve1d(
#     frames,
#     g_space,
#     axis=2,
#     mode="reflect"
# )
# ### note that the paper does not tell us about the boundry frames how to convolve there...

# # Gaussian smoothing in y

# G = convolve1d(
#     G,
#     g_space,
#     axis=1,
#     mode="reflect"
# )


# # Gaussian smoothing in time

# G = convolve1d(
#     G,
#     g_time,
#     axis=0,
#     mode="reflect"
# )


# ------------------------------------------------------------
# Second derivative with respect to x
# ------------------------------------------------------------

# G_xx = convolve1d(
#     frames,
#     g_space_xx,
#     axis=2,
#     mode="reflect"
# )

# G_xx = convolve1d(
#     G_xx,
#     g_space,
#     axis=1,
#     mode="reflect"
# )

# G_xx = convolve1d(
#     G_xx,
#     g_time,
#     axis=0,
#     mode="reflect"
# )


# # ------------------------------------------------------------
# # Second derivative with respect to y
# # ------------------------------------------------------------

# G_yy = convolve1d(
#     frames,
#     g_space,
#     axis=2,
#     mode="reflect"
# )

# G_yy = convolve1d(
#     G_yy,
#     g_space_xx,
#     axis=1,
#     mode="reflect"
# )

# G_yy = convolve1d(
#     G_yy,
#     g_time,
#     axis=0,
#     mode="reflect"
# )


# # ------------------------------------------------------------
# # Second derivative with respect to time
# # ------------------------------------------------------------

# G_tt = convolve1d(
#     frames,
#     g_space,
#     axis=2,
#     mode="reflect"
# )

# G_tt = convolve1d(
#     G_tt,
#     g_space,
#     axis=1,
#     mode="reflect"
# )

# G_tt = convolve1d(
#     G_tt,
#     g_time_tt,
#     axis=0,
#     mode="reflect"
# )


# ============================================================
# STEP 6: DALEMBERTIAN / MASK RESPONSE
# ============================================================

# Paper:
#
# m(x,y,t) =
# -( ∇² + (1/u²) ∂²/∂t² ) G(x,y,t)
#
# Since:
#
# ∇²G = G_xx + G_yy
#
# Therefore:
#
# S = -(G_xx + G_yy + G_tt/u²)

# S = -(G_xx + G_yy + G_tt / (u ** 2))

# print("Edge signal calculated.")
#################################################################





#################################################################
# ============================================================
# STEP 5: CONSTRUCT THE 3-D MASK / KERNEL m
# ============================================================

# Array ordering:
#
# frames[time, y, x]
#
# Therefore the 3-D kernels have ordering:
#
# (time, y, x)


# ------------------------------------------------------------
# x-second-derivative term
# ------------------------------------------------------------

# g''(x) * g(y) * g(t)

K_xx = (
    g_time[:, None, None]
    * g_space[None, :, None]
    * g_space_xx[None, None, :]
)


# ------------------------------------------------------------
# y-second-derivative term
# ------------------------------------------------------------

# g(x) * g''(y) * g(t)

K_yy = (
    g_time[:, None, None]
    * g_space_xx[None, :, None]
    * g_space[None, None, :]
)


# ------------------------------------------------------------
# time-second-derivative term
# ------------------------------------------------------------

# g(x) * g(y) * g''(t)

K_tt = (
    g_time_tt[:, None, None]
    * g_space[None, :, None]
    * g_space[None, None, :]
)


# ============================================================
# STEP 6: FINAL 3-D MASK / KERNEL m
# ============================================================

# Paper:
#
# m(x,y,t) =
# -( ∇² + (1/u²) ∂²/∂t² ) G(x,y,t)
#
# where
#
# ∇² = ∂²/∂x² + ∂²/∂y²
#
# Therefore:
#
# m =
# -(K_xx + K_yy + K_tt/u²)

m = -(
    K_xx
    + K_yy
    + K_tt / (u ** 2)
)

print("3-D mask m calculated.")


# ============================================================
# STEP 7: CONVOLVE THE MASK WITH THE ORIGINAL VIDEO
# ============================================================


S = convolve(
    frames,
    m,
    mode="reflect"
)

print("Final edge signal S calculated.")


# ============================================================
# STEP 8: FIND ZERO-CROSSING CANDIDATES
# ============================================================

def find_zero_crossings(S):#it means define a function find_... that takes S as input 
    """
    Find candidate zero crossings in the 3-D edge signal S.

    Array ordering:
        S[k, j, i]

        k -> time
        j -> y
        i -> x

    A zero crossing is detected when neighboring values
    have opposite signs:

        S1 * S2 < 0
    """

    zero_crossings = []

    nt, ny, nx = S.shape


    # ========================================================
    # ZERO CROSSINGS IN X DIRECTION
    # ========================================================

    for k in range(nt):

        for j in range(ny):

            for i in range(nx - 1):

                left = S[k, j, i]
                right = S[k, j, i + 1]

                if left * right < 0:

                    zero_crossings.append(
                        (k, j, i, "x")
                    )


    # ========================================================
    # ZERO CROSSINGS IN Y DIRECTION
    # ========================================================

    for k in range(nt):

        for j in range(ny - 1):

            for i in range(nx):

                current = S[k, j, i]
                next_value = S[k, j + 1, i]

                if current * next_value < 0:

                    zero_crossings.append(
                        (k, j, i, "y")
                    )


    # ========================================================
    # ZERO CROSSINGS IN TIME DIRECTION
    # ========================================================

    for k in range(nt - 1):

        for j in range(ny):

            for i in range(nx):

                previous = S[k, j, i]
                next_value = S[k + 1, j, i]

                if previous * next_value < 0:

                    zero_crossings.append(
                        (k, j, i, "t")
                    )


    return zero_crossings


# ============================================================
# GET ZERO-CROSSING CANDIDATES
# ============================================================

zero_crossings = find_zero_crossings(S)

print(
    "Number of zero-crossing candidates:",
    len(zero_crossings)
)

##################################################################################################################################################
# ============================================================
# STEP 8: FIND ZERO-CROSSING CANDIDATES
# ============================================================

def find_zero_crossings(S, EDGE_DIRECTION):
    """
    Find spatial zero-crossing candidates in the 3-D edge signal S.

    Array ordering:
        S[k, j, i]

        k -> time
        j -> y
        i -> x

    For a vertical edge:
        Search row-wise along x.

    For a horizontal edge:
        Search column-wise along y.

    The zero-crossing candidate is identified by checking
    the values on either side of the candidate location:

        S(i-1) and S(i+1)     for x-direction

        S(j-1) and S(j+1)     for y-direction

    A zero crossing is detected when these values
    have opposite signs.
    """

    zero_crossings = []

    nt, ny, nx = S.shape


    # ========================================================
    # VERTICAL EDGE
    #
    # Search row-wise along x
    # ========================================================

    if EDGE_DIRECTION.lower() == "vertical":

        for k in range(nt):

            for j in range(ny):

                for i in range(1, nx - 1):

                    left = S[k, j, i - 1]

                    right = S[k, j, i + 1]

                    if left * right < 0:

                        zero_crossings.append(
                            (k, j, i, "x")
                        )


    # ========================================================
    # HORIZONTAL EDGE
    #
    # Search column-wise along y
    # ========================================================

    elif EDGE_DIRECTION.lower() == "horizontal":

        for k in range(nt):

            for i in range(nx):

                for j in range(1, ny - 1):

                    previous = S[k, j - 1, i]

                    next_value = S[k, j + 1, i]

                    if previous * next_value < 0:

                        zero_crossings.append((k, j, i, "y"))


    # ========================================================
    # INVALID EDGE DIRECTION
    # ========================================================

    else:

        raise ValueError("EDGE_DIRECTION must be " "'vertical' or 'horizontal'")


    return zero_crossings


# ============================================================
# GET ZERO-CROSSING CANDIDATES
# ============================================================

zero_crossings = find_zero_crossings(S,EDGE_DIRECTION)

print("Number of zero-crossing candidates:",len(zero_crossings))
# ============================================================
# STEP 9: SUB-PIXEL ZERO-CROSSING
# ============================================================

def subpixel_zero_crossing(neighborhood, direction):
    """
    Estimate the sub-pixel spatial zero crossing using the
    local 3-D linear hyperplane:

        f(i,j,k) = f0 + fx*i + fy*j + fz*k

    Array ordering:

        neighborhood[k, j, i]

        k -> time
        j -> y
        i -> x

    direction:
        "x" -> zero crossing along x
        "y" -> zero crossing along y

    Time is used in the hyperplane fitting through fz,
    but is NOT used as a zero-crossing direction.

    Returns:
        sub-pixel offset in the detected spatial direction
    """


    # --------------------------------------------------------
    # Get neighborhood dimensions
    # --------------------------------------------------------

    nt, ny, nx = neighborhood.shape


    # --------------------------------------------------------
    # Create local coordinates
    #
    # For a 3 × 3 × 3 neighborhood:
    #
    # k = [-1, 0, +1]
    # j = [-1, 0, +1]
    # i = [-1, 0, +1]
    # --------------------------------------------------------

    k, j, i = np.meshgrid(
        np.arange(-(nt // 2), nt // 2 + 1),
        np.arange(-(ny // 2), ny // 2 + 1),
        np.arange(-(nx // 2), nx // 2 + 1),
        indexing="ij"
    )


    # --------------------------------------------------------
    # Flatten the 3-D neighborhood
    #
    # 3 × 3 × 3 = 27 values
    # --------------------------------------------------------

    S_values = neighborhood.ravel()

    i = i.ravel()
    j = j.ravel()
    k = k.ravel()


    f0 = np.sum(S_values) / 27

    fx = np.sum(i * S_values) / 18

    fy = np.sum(j * S_values) / 18

    fz = np.sum(k * S_values) / 18


    # --------------------------------------------------------
    # Calculate spatial zero-crossing position
    # only in the detected direction
    # --------------------------------------------------------

    if direction == "x":

        if fx != 0:

            offset = -f0 / fx

        else:

            offset = np.nan


    elif direction == "y":

        if fy != 0:

            offset = -f0 / fy

        else:

            offset = np.nan


    else:

        raise ValueError("direction must be 'x' or 'y'")


    return offset


# ============================================================
# APPLY STEP 9 TO ALL ZERO-CROSSING CANDIDATES
# ============================================================

subpixel_results = []


for k, j, i, direction in zero_crossings:

    # --------------------------------------------------------
    # Extract the local 3 × 3 × 3 neighborhood
    # --------------------------------------------------------

    neighborhood = S[ k - 1:k + 2,j - 1:j + 2, i - 1:i + 2]


    # --------------------------------------------------------
    # Make sure a complete 3 × 3 × 3 neighborhood exists
    # --------------------------------------------------------

    if neighborhood.shape != (3, 3, 3):

        continue


    # --------------------------------------------------------
    # Calculate the sub-pixel zero-crossing offset
    # --------------------------------------------------------

    offset = subpixel_zero_crossing(neighborhood,direction)


    # --------------------------------------------------------
    # Store the result
    # --------------------------------------------------------

    subpixel_results.append((k,j,i,direction,offset))


print("Number of sub-pixel zero-crossing results:",len(subpixel_results))
###############################################################################################################################################
# ============================================================
# STEP 10: CONSTRUCT EDGE-POSITION / DISPLACEMENT MATRIX
# ============================================================

T, H, W = S.shape

print("\nConstructing edge-position matrix...")


# ============================================================
# VERTICAL STRUCTURAL EDGE
# ============================================================

if EDGE_DIRECTION.lower() == "vertical":

    # --------------------------------------------------------
    # For a vertical edge:
    #
    # The edge position is measured along x.
    #
    # displacement[row, time]
    # --------------------------------------------------------

    displacement = np.full((H, T),np.nan)


    # --------------------------------------------------------
    # Process the zero-crossing results obtained from
    # Steps 8 and 9
    # --------------------------------------------------------

    for k, j, i, direction, offset in subpixel_results:

        # Make sure this is an x-direction crossing
        if direction != "x":
            continue

        # Make sure the offset is valid
        if not np.isfinite(offset):
            continue

        # ----------------------------------------------------
        # Convert sub-pixel offset into actual edge position
        #
        # i      -> integer pixel position
        # offset -> sub-pixel correction
        # ----------------------------------------------------

        position = i + offset


        # ----------------------------------------------------
        # Store the edge position
        #
        # If more than one crossing occurs for the same
        # row and time, keep the strongest crossing.
        # ----------------------------------------------------

        if np.isnan(displacement[j, k]):

            displacement[j, k] = position

        else:

            # Existing edge position
            existing_position = displacement[j, k]

            # Integer locations corresponding to the two
            # candidate positions
            c_new = int(round(position))
            c_old = int(round(existing_position))

            # Make sure both locations are valid
            if (
                1 <= c_new < W - 1
                and 1 <= c_old < W - 1
            ):

                # Signal at this time and row
                signal = S[k, j, :]

                # Gradient of new candidate
                gradient_new = abs(
                    signal[c_new + 1]
                    - signal[c_new - 1]
                )

                # Gradient of existing candidate
                gradient_old = abs(
                    signal[c_old + 1]
                    - signal[c_old - 1]
                )

                # Keep the stronger crossing
                if gradient_new > gradient_old:

                    displacement[j, k] = position


# ============================================================
# HORIZONTAL STRUCTURAL EDGE
# ============================================================

elif EDGE_DIRECTION.lower() == "horizontal":

    # --------------------------------------------------------
    # For a horizontal edge:
    #
    # The edge position is measured along y.
    #
    # displacement[col, time]
    # --------------------------------------------------------

    displacement = np.full(
        (W, T),
        np.nan
    )


    # --------------------------------------------------------
    # Process the zero-crossing results obtained from
    # Steps 8 and 9
    # --------------------------------------------------------

    for k, j, i, direction, offset in subpixel_results:

        # Make sure this is a y-direction crossing
        if direction != "y":
            continue

        # Make sure the offset is valid
        if not np.isfinite(offset):
            continue

        # ----------------------------------------------------
        # Convert sub-pixel offset into actual edge position
        #
        # j      -> integer pixel position
        # offset -> sub-pixel correction
        # ----------------------------------------------------

        position = j + offset


        # ----------------------------------------------------
        # Store the edge position
        #
        # If more than one crossing occurs for the same
        # column and time, keep the strongest crossing.
        # ----------------------------------------------------

        if np.isnan(displacement[i, k]):

            displacement[i, k] = position

        else:

            # Existing edge position
            existing_position = displacement[i, k]

            # Integer locations corresponding to the two
            # candidate positions
            r_new = int(round(position))
            r_old = int(round(existing_position))

            # Make sure both locations are valid
            if (
                1 <= r_new < H - 1
                and 1 <= r_old < H - 1
            ):

                # Signal at this time and column
                signal = S[k, :, i]

                # Gradient of new candidate
                gradient_new = abs(
                    signal[r_new + 1]
                    - signal[r_new - 1]
                )

                # Gradient of existing candidate
                gradient_old = abs(
                    signal[r_old + 1]
                    - signal[r_old - 1]
                )

                # Keep the stronger crossing
                if gradient_new > gradient_old:

                    displacement[i, k] = position


# ============================================================
# INVALID EDGE DIRECTION
# ============================================================

else:

    raise ValueError("EDGE_DIRECTION must be " "'vertical' or 'horizontal'")


print("Edge-position matrix constructed:",displacement.shape)

