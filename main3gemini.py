import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.ndimage import convolve, gaussian_filter
from scipy.linalg import svd as scipy_svd

# ============================================================
# USER SETTINGS
# ============================================================

VIDEO_PATH = "optical_flow_vdo.mp4"

# Paper used wm = 5 and wt = 3 for the cantilever experiment
WM = 5.0
WT = 3.0

# Choose orientation: "vertical" or "horizontal"
EDGE_DIRECTION = "vertical"

# Which row/column to use for single-point validation plot
REFERENCE_INDEX = None

# Pixel-to-mm conversion factor
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

sigma_space = WM / (2.0 * np.sqrt(2.0))
u = WM / WT
sigma_time = sigma_space / u

print("\nFilter parameters")
print("-----------------")
print("Spatial sigma   :", sigma_space)
print("Velocity scale u:", u)
print("Temporal sigma  :", sigma_time)


# ============================================================
# STEP 5: 3-D GAUSSIAN DERIVATIVES & MASK CONSTRUCTION
# ============================================================

spatial_radius = int(np.ceil(3.0 * sigma_space))
x_vec = np.arange(-spatial_radius, spatial_radius + 1, dtype=np.float64)

g_space = (1.0 / (sigma_space * np.sqrt(2.0 * np.pi))) * np.exp(-x_vec**2 / (2.0 * sigma_space**2))

g_space_xx = (
    (1.0 / (sigma_space**3 * np.sqrt(2.0 * np.pi)))
    * (x_vec**2 / sigma_space**2 - 1.0)
    * np.exp(-x_vec**2 / (2.0 * sigma_space**2))
)

temporal_radius = int(np.ceil(3.0 * sigma_time))
t_vec = np.arange(-temporal_radius, temporal_radius + 1, dtype=np.float64)

g_time = (1.0 / (sigma_time * np.sqrt(2.0 * np.pi))) * np.exp(-t_vec**2 / (2.0 * sigma_time**2))

g_time_tt = (
    (1.0 / (sigma_time**3 * np.sqrt(2.0 * np.pi)))
    * (t_vec**2 / sigma_time**2 - 1.0)
    * np.exp(-t_vec**2 / (2.0 * sigma_time**2))
)

# 3D Kernel Components: order (time, y, x)
K_xx = g_time[:, None, None] * g_space[None, :, None] * g_space_xx[None, None, :]
K_yy = g_time[:, None, None] * g_space_xx[None, :, None] * g_space[None, None, :]
K_tt = g_time_tt[:, None, None] * g_space[None, :, None] * g_space[None, None, :]

# Mask m(x,y,t) = -(K_xx + K_yy + K_tt / u^2)
m = -(K_xx + K_yy + K_tt / (u**2))
print("3-D mask m calculated.")


# ============================================================
# STEP 6: CONVOLVE MASK WITH ORIGINAL VIDEO
# ============================================================

S = convolve(frames, m, mode="reflect")
print("Final edge signal S calculated.")


# ============================================================
# STEP 7: SUB-PIXEL ZERO-CROSSING FUNCTION
# ============================================================

def subpixel_zero_crossing(neighborhood):
    """
    Computes sub-pixel zero crossing offset using 3x3x3 hyperplane fit.
    Paper Appendix Eqs. (10)-(15).
    """
    nt, ny, nx = neighborhood.shape
    if nt != 3 or ny != 3 or nx != 3:
        return np.nan, np.nan, np.nan

    f0 = np.mean(neighborhood)

    # Gradients along x, y, t axes
    i_coords = np.array([-1, 0, 1])
    fx = np.sum(i_coords[None, None, :] * neighborhood) / 18.0
    fy = np.sum(i_coords[None, :, None] * neighborhood) / 18.0
    fz = np.sum(i_coords[:, None, None] * neighborhood) / 18.0

    x_offset = -f0 / fx if abs(fx) > 1e-12 else np.nan
    y_offset = -f0 / fy if abs(fy) > 1e-12 else np.nan
    t_offset = -f0 / fz if abs(fz) > 1e-12 else np.nan

    return x_offset, y_offset, t_offset


# ============================================================
# STEP 8: TRACK EDGE POSITIONS
# ============================================================

T, H, W = S.shape
print("\nTracking edge...")

if EDGE_DIRECTION.lower() == "vertical":
    displacement = np.full((H, T), np.nan)

    for t in range(1, T - 1):
        for row in range(1, H - 1):
            signal = S[t, row, :]
            crossings = []

            for col in range(1, W - 1):
                if signal[col] * signal[col + 1] <= 0:
                    neighborhood = S[t - 1:t + 2, row - 1:row + 2, col - 1:col + 2]
                    x_offset, _, _ = subpixel_zero_crossing(neighborhood)

                    if np.isfinite(x_offset) and abs(x_offset) <= 1.5:
                        position = col + x_offset
                        crossings.append((position, col))

            if len(crossings) > 0:
                best_pos = None
                best_gradient = -np.inf
                for pos, c in crossings:
                    if 1 <= c < W - 1:
                        gradient = abs(signal[c + 1] - signal[c - 1])
                        if gradient > best_gradient:
                            best_gradient = gradient
                            best_pos = pos
                displacement[row, t] = best_pos

elif EDGE_DIRECTION.lower() == "horizontal":
    displacement = np.full((W, T), np.nan)

    for t in range(1, T - 1):
        for col in range(1, W - 1):
            signal = S[t, :, col]
            crossings = []

            for row in range(1, H - 1):
                if signal[row] * signal[row + 1] <= 0:
                    neighborhood = S[t - 1:t + 2, row - 1:row + 2, col - 1:col + 2]
                    _, y_offset, _ = subpixel_zero_crossing(neighborhood)

                    if np.isfinite(y_offset) and abs(y_offset) <= 1.5:
                        position = row + y_offset
                        crossings.append((position, row))

            if len(crossings) > 0:
                best_pos = None
                best_gradient = -np.inf
                for pos, r in crossings:
                    if 1 <= r < H - 1:
                        gradient = abs(signal[r + 1] - signal[r - 1])
                        if gradient > best_gradient:
                            best_gradient = gradient
                            best_pos = pos
                displacement[col, t] = best_pos


# ============================================================
# STEP 9: INTERPOLATE & CLEAN MISSING VALUES BEFORE RPCA
# ============================================================

D = displacement.copy()

# Fill NaNs via pandas interpolation
df_D = pd.DataFrame(D)
df_D = df_D.interpolate(method='linear', axis=1).interpolate(method='linear', axis=0).ffill().bfill()
D = df_D.to_numpy()

# Ensure absolute numeric stability before matrix factorization
D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# STEP 10: ROBUST PCA / IALM WITH STABLE SVD
# ============================================================

def robust_pca_ialm(D, lam=None, tol=1e-7, max_iter=1000):
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
    m, n = D.shape

    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    L = np.zeros_like(D)
    E = np.zeros_like(D)
    Y = np.zeros_like(D)

    norm_D = np.linalg.norm(D, ord='fro')
    if norm_D < 1e-12:
        return D.copy(), np.zeros_like(D)

    mu = 1.25 / (norm_D + 1e-7)
    mu_bar = mu * 1e7
    rho = 1.5

    def singular_value_threshold(X, tau):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback to LAPACK 'gesvd' driver on convergence failure
            U, s, Vt = scipy_svd(X, full_matrices=False, lapack_driver='gesvd')

        s_thresholded = np.maximum(s - tau, 0)
        rank = np.sum(s_thresholded > 0)
        if rank == 0:
            return np.zeros_like(X)
        return U[:, :rank] @ np.diag(s_thresholded[:rank]) @ Vt[:rank, :]

    def soft_threshold(X, tau):
        return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

    for iteration in range(max_iter):
        L = singular_value_threshold(D - E + (Y / mu), 1.0 / mu)
        E = soft_threshold(D - L + (Y / mu), lam / mu)
        residual = D - L - E
        Y = Y + mu * residual
        mu = min(rho * mu, mu_bar)

        error = np.linalg.norm(residual, ord='fro') / norm_D
        if error < tol:
            print(f"IALM converged at iteration {iteration + 1}")
            break

    return L, E


print("\nApplying Robust PCA / IALM...")
L, E = robust_pca_ialm(D)
print("RPCA completed.")


# ============================================================
# STEP 11: 2-D GAUSSIAN SMOOTHING OF LOW-RANK COMPONENT
# ============================================================

displacement_clean = scipy.ndimage.gaussian_filter(
    L,
    sigma=(sigma_space, sigma_time),
    mode="nearest"
)


# ============================================================
# STEP 12: CONVERT TO LAGRANGIAN DISPLACEMENT
# ============================================================

for i in range(displacement_clean.shape[0]):
    initial_position = displacement_clean[i, 0]
    displacement_clean[i, :] -= initial_position


# ============================================================
# STEP 13 & 14: PHYSICAL UNITS & TIME VECTOR
# ============================================================

displacement_physical = displacement_clean * PIXEL_TO_MM
time = np.arange(T) / fps


# ============================================================
# STEP 15: SAVE FULL-FIELD DISPLACEMENT
# ============================================================

data = {"time_s": time}
num_points = displacement_physical.shape[0]

for i in range(num_points):
    key = f"row_{i}_disp_mm" if EDGE_DIRECTION.lower() == "vertical" else f"column_{i}_disp_mm"
    data[key] = displacement_physical[i, :]

df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved full-field displacements to: {OUTPUT_CSV}")


# ============================================================
# STEP 16-18: PLOTTING
# ============================================================

if REFERENCE_INDEX is None:
    REFERENCE_INDEX = displacement_physical.shape[0] // 2

reference_signal = displacement_physical[REFERENCE_INDEX, :]

plt.figure(figsize=(10, 4))
plt.plot(time, reference_signal, linewidth=1.2)
plt.xlabel("Time (s)")
plt.ylabel("Displacement (mm)")
plt.title("Single-Point Lagrangian Displacement Response")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(
    displacement_physical,
    aspect="auto",
    extent=[time[0], time[-1], 0, displacement_physical.shape[0]],
    origin="lower"
)
plt.colorbar(label="Displacement (mm)")
plt.xlabel("Time (s)")
plt.ylabel("Position along edge (pixel)")
plt.title("Full-Field Displacement Time History")
plt.tight_layout()
plt.show()

print("\nProcessing completed successfully.")