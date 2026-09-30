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

K_tt = (g_time_tt[:, None, None]* g_space[None, :, None]* g_space[None, None, :])


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

m = -(K_xx+K_yy+K_tt / (u ** 2))

print("3-D mask m calculated.")


# ============================================================
# STEP 7: CONVOLVE THE MASK WITH THE ORIGINAL VIDEO
# ============================================================
from scipy.ndimage import convolve1d
from scipy.ndimage import convolve

S = convolve(frames, m,mode="reflect")

print("Final edge signal S calculated.")


# ============================================================
# STEP 8: FIND ZERO-CROSSING CANDIDATES
# ============================================================

def find_zero_crossings(S):
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

                    zero_crossings.append((k, j, i, "x"))

    # ========================================================
    # ZERO CROSSINGS IN Y DIRECTION
    # ========================================================

    for k in range(nt):

        for j in range(ny - 1):

            for i in range(nx):

                current = S[k, j, i]
                next_value = S[k, j + 1, i]

                if current * next_value < 0:

                    zero_crossings.append((k, j, i, "y"))

    return zero_crossings

# ============================================================
# GET ZERO-CROSSING CANDIDATES
# ============================================================

zero_crossings = find_zero_crossings(S)

print("Number of zero-crossing candidates:",len(zero_crossings))

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

                        zero_crossings.append((k, j, i, "x"))


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
        indexing="ij")


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
# STEP 10: CONSTRUCT EDGE-POSITION MATRIX
# ============================================================

T, H, W = S.shape

print("\nConstructing sub-pixel position matrix...")


# ============================================================
# VERTICAL EDGE ONLY
# ============================================================

if EDGE_DIRECTION.lower() == "vertical":

    # --------------------------------------------------------
    # VERTICAL EDGE
    #
    # Edge position is measured along x.
    #
    # Rows    -> spatial y-positions
    # Columns -> time
    #
    # d = i + offset       if sub-pixel is available
    # d = i                 if sub-pixel is not available
    #
    # i      -> integer pixel coordinate
    # offset -> sub-pixel correction
    # d      -> final pixel/sub-pixel x-coordinate
    # --------------------------------------------------------

    # Temporary full ROI matrix
    D_position_v = np.full((H, T), np.nan)

    # --------------------------------------------------------
    # Fill detected edge positions
    # --------------------------------------------------------

    for k, j, i, direction, offset in subpixel_results:

        # We only want vertical-edge detections
        if direction != "x":
            continue

        # ----------------------------------------------------
        # Determine edge position d
        # ----------------------------------------------------

        if np.isfinite(offset):
            d = i + offset
        else:
            d = i

        # ----------------------------------------------------
        # Store edge position
        #
        # j -> spatial y-position
        # k -> time
        #
        # D_position_v[j,k] = x-coordinate of edge
        # ----------------------------------------------------

        D_position_v[j, k] = d

    # --------------------------------------------------------
    # Remove spatial positions where NO edge was detected
    # during the entire video
    # --------------------------------------------------------

    valid_positions = np.any(
        np.isfinite(D_position_v),
        axis=1
    )

    # --------------------------------------------------------
    # FINAL MEASUREMENT MATRIX
    #
    # This is D.
    #
    # D = measured vertical edge-position matrix
    # --------------------------------------------------------

    D = D_position_v[valid_positions, :]

    # Retain original y-positions
    spatial_positions = np.where(valid_positions)[0]

    print(
        "Pixel/sub-pixel edge-position matrix constructed."
    )

    print(
        "D shape:",
        D.shape
    )

    print(
        "Number of spatial positions:",
        len(spatial_positions)
    )


# ============================================================
# DO NOT PROCESS HORIZONTAL EDGE
# ============================================================

else:

    raise ValueError(
        "This code is configured for the vertical edge only. "
        "Set EDGE_DIRECTION = 'vertical'."
    )


# ============================================================
# STEP 11: IALM / RPCA FUNCTION
# ============================================================

def ialm_rpca(D, max_iter=1000, tol=1e-7):
    """
    Robust Principal Component Analysis using
    Inexact Augmented Lagrange Multipliers (IALM).

    Decomposes:

        D = L + E

    where:

        D -> measured pixel/sub-pixel POSITION matrix
        L -> low-rank CLEAN POSITION matrix
        E -> sparse ERROR matrix

    D shape:
        spatial position x time
    """

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not np.all(np.isfinite(D)):
        raise ValueError(
            "D contains NaN or infinite values. "
            "IALM requires a complete measurement matrix."
        )

    D = np.asarray(D, dtype=float)

    m, n = D.shape

    # --------------------------------------------------------
    # Regularization parameter
    # --------------------------------------------------------

    lam = 1.0 / np.sqrt(max(m, n))

    # --------------------------------------------------------
    # Initial values
    # --------------------------------------------------------

    L = np.zeros_like(D)
    E = np.zeros_like(D)
    Y = np.zeros_like(D)

    # --------------------------------------------------------
    # Initial penalty parameter
    # --------------------------------------------------------

    mu = (
        (m * n)
        /
        (4.0 * np.sum(np.abs(D)) + 1e-12)
    )

    mu_bar = mu * 1e7
    rho = 1.5

    # --------------------------------------------------------
    # IALM iterations
    # --------------------------------------------------------

    for iteration in range(max_iter):

        # ====================================================
        # STEP 1: UPDATE LOW-RANK MATRIX L
        #         Singular Value Thresholding
        # ====================================================

        U_svd, singular_values, Vt = np.linalg.svd(
            D - E + Y / mu,
            full_matrices=False
        )

        threshold = 1.0 / mu

        singular_values_thresholded = np.maximum(
            singular_values - threshold,
            0
        )

        L = (
            U_svd
            @
            np.diag(singular_values_thresholded)
            @
            Vt
        )

        # ====================================================
        # STEP 2: UPDATE SPARSE ERROR MATRIX E
        #         Soft Thresholding
        # ====================================================

        temp = D - L + Y / mu

        E = (
            np.sign(temp)
            *
            np.maximum(
                np.abs(temp) - lam / mu,
                0
            )
        )

        # ====================================================
        # STEP 3: UPDATE LAGRANGE MULTIPLIER
        # ====================================================

        residual = D - L - E

        Y = Y + mu * residual

        # ====================================================
        # STEP 4: INCREASE PENALTY PARAMETER
        # ====================================================

        mu = min(
            rho * mu,
            mu_bar
        )

        # ====================================================
        # STEP 5: CONVERGENCE CHECK
        # ====================================================

        error = (
            np.linalg.norm(residual, "fro")
            /
            (
                np.linalg.norm(D, "fro")
                +
                1e-12
            )
        )

        if error < tol:

            print(
                f"IALM converged at iteration "
                f"{iteration + 1}"
            )

            break

    else:

        print(
            "IALM reached maximum iterations "
            "without reaching tolerance."
        )

    return L, E


# ============================================================
# STEP 12: GAUSSIAN SMOOTHING
# ============================================================

def smooth_displacement(U, WM, WT):
    """
    Apply 2-D Gaussian smoothing to the
    displacement matrix.

    U shape:
        spatial position x time

    Axis 0:
        spatial direction

    Axis 1:
        temporal direction
    """

    # --------------------------------------------------------
    # Convert mask widths to Gaussian standard deviations
    # --------------------------------------------------------

    sigma_space = WM / (
        2.0 * np.sqrt(2.0)
    )

    sigma_time = WT / (
        2.0 * np.sqrt(2.0)
    )

    # --------------------------------------------------------
    # Apply 2-D Gaussian smoothing
    #
    # axis 0 -> spatial position
    # axis 1 -> time
    # --------------------------------------------------------

    U_smooth = gaussian_filter(
        U,
        sigma=(
            sigma_space,
            sigma_time
        ),
        mode="nearest"
    )

    return U_smooth


# ============================================================
# STEP 13: RPCA / IALM DECOMPOSITION
#
# D = L + E
# ============================================================

print(
    "\nProcessing vertical edge-position matrix "
    "using RPCA / IALM..."
)

# ------------------------------------------------------------
# Decompose D
#
# D -> measured position
# L -> clean position
# E -> sparse error
# ------------------------------------------------------------

L, E = ialm_rpca(D)

print(
    "D shape:",
    D.shape
)

print(
    "Low-rank L shape:",
    L.shape
)

print(
    "Sparse error E shape:",
    E.shape
)


# ============================================================
# STEP 14: CALCULATE VERTICAL DISPLACEMENT
# ============================================================

# ------------------------------------------------------------
# Reference = position at the first video frame
#
# Every spatial point has its own reference position:
#
# U(y,t) =
#
#       L(y,t)
#       -
#       L(y,t0)
# ------------------------------------------------------------

reference_position = L[:, 0]

U = (
    L
    -
    reference_position[:, None]
)

print(
    "Vertical displacement matrix calculated."
)

print(
    "U shape:",
    U.shape
)


# ============================================================
# STEP 15: GAUSSIAN SMOOTHING OF DISPLACEMENT
# ============================================================

U_smooth = smooth_displacement(
    U,
    WM,
    WT
)

print(
    "Gaussian smoothing of vertical "
    "displacement completed."
)

print(
    "U_smooth shape:",
    U_smooth.shape
)


# ============================================================
# STEP 16: PLOT FINAL VERTICAL DISPLACEMENT
# ============================================================

plt.figure(figsize=(10, 6))

plt.imshow(
    U_smooth,
    aspect="auto",
    origin="lower",
    cmap="jet"
)

plt.xlabel(
    "Time frame"
)

plt.ylabel(
    "Spatial position"
)

plt.title(
    "Full-Field Vertical Displacement"
)

plt.colorbar(
    label="Displacement (pixels)"
)

plt.tight_layout()

plt.show()