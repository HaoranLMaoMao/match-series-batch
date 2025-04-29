import os
import numpy as np
import hyperspy.api as hs
from tqdm import tqdm
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import view_as_windows
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from PIL import Image
from .utils import extract_number, write_log
from pymatchseries import MatchSeries

def extract_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Extracts all overlapping patches of size (patch_size x patch_size)
    from a 2D image and flattens them into shape (num_patches, patch_size^2).
    """
    patches = view_as_windows(image, (patch_size, patch_size))
    return patches.reshape(-1, patch_size * patch_size)

def reconstruct_from_patches(patches_denoised: np.ndarray,
                             image_shape: tuple,
                             patch_size: int,
                             idxs: np.ndarray) -> np.ndarray:
    """
    Reconstructs a denoised image from its denoised patches.
    Each patch is added back into its original location, and we
    average overlapping contributions.
    """
    recon = np.zeros(image_shape, dtype=float)
    weight = np.zeros(image_shape, dtype=float)
    # coords of top‐left corner for each patch
    coords = np.argwhere(np.ones((image_shape[0] - patch_size + 1,
                                  image_shape[1] - patch_size + 1)))
    for k, patch_index in enumerate(idxs):
        i, j = coords[patch_index]
        recon[i:i+patch_size, j:j+patch_size] += patches_denoised[k]
        weight[i:i+patch_size, j:j+patch_size] += 1
    # avoid division by zero
    return recon / np.maximum(weight, 1)

def nlpca_denoise(image: np.ndarray,
                  patch_size: int,
                  n_clusters: int,
                  n_components: int,
                  method: str) -> np.ndarray:
    """
    Performs non-local PCA denoising via clustering + NMF.
    method: 'nlpca-kmeans', 'nlpca-spectral', or 'nlpca-gmm'
    """
    # 1. Extract patches
    patches = extract_patches(image, patch_size)

    # 2. Cluster patches
    if method == 'nlpca-kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        labels = clusterer.fit_predict(patches)
    elif method == 'nlpca-spectral':
        clusterer = SpectralClustering(
            n_clusters=n_clusters,
            assign_labels='kmeans',
            affinity='nearest_neighbors',
            n_neighbors=10,
            random_state=0,
            n_jobs=-1
        )
        labels = clusterer.fit_predict(patches)
    elif method == 'nlpca-gmm':
        clusterer = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            random_state=0
        )
        labels = clusterer.fit_predict(patches)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # 3. For each cluster, apply NMF to its patches
    recon = np.zeros(image.shape, dtype=float)
    weight = np.zeros(image.shape, dtype=float)
    for c in range(n_clusters):
        idxs = np.where(labels == c)[0]
        if idxs.size == 0:
            continue
        patch_group = patches[idxs]

        # Non-negative matrix factorization
        nmf = NMF(n_components=n_components,
                  beta_loss='kullback-leibler',
                  solver='mu',
                  max_iter=300)
        W = nmf.fit_transform(patch_group)
        H = nmf.components_
        denoised_patches = (W @ H).reshape(-1, patch_size, patch_size)

        # Reconstruct partial image
        partial = reconstruct_from_patches(
            denoised_patches, image.shape, patch_size, idxs
        )
        recon += partial
        weight += (weight == 0)  # increment weight mask

    # 4. Normalize
    return recon / np.maximum(weight, 1)

def process_one_sample(sample_name: str,
                       input_folder: str,
                       output_folder: str,
                       log_file_path: str,
                       regularization_lambda: float = 20,
                       filename_prefix: str = "Aligned_",
                       save_dtype: str = 'uint8',
                       denoising_method: str = 'nlmeans',
                       nlpca_patch_size: int = 7,
                       nlpca_n_clusters: int = 10,
                       nlpca_n_components: int = 8):
    """
    Performs the full pipeline on one sample:
     1. Non-rigid registration (pyMatchSeries)
     2. Save each frame as TIFF + full aligned stack as HSPY
     3. Compute stage‐average image
     4. Optionally denoise the average using:
        - 'none'          : no denoising
        - 'nlmeans'       : non-local means
        - 'nlpca-<method>': clustering + NMF
     5. Save denoised average as TIFF + HSPY
    Logs progress and errors to log_file_path.
    """
    from pymatchseries import MatchSeries

    os.makedirs(output_folder, exist_ok=True)

    # 1) Gather .dm4 files
    file_list = sorted(
        [f for f in os.listdir(input_folder)
         if f.endswith(".dm4") and not f.startswith("._")],
        key=extract_number
    )
    if not file_list:
        write_log(log_file_path, f"No .dm4 files found in {input_folder}.")
        return

    # 2) Load images
    images = []
    for fname in file_list:
        path = os.path.join(input_folder, fname)
        try:
            sig = hs.load(path, lazy=True)
            images.append(sig)
        except Exception as e:
            write_log(log_file_path, f"Failed to load {fname}: {e}")

    if not images:
        write_log(log_file_path, f"No valid images in {input_folder}.")
        return

    # 3) Stack and configure MatchSeries
    stack = hs.stack(images)
    match = MatchSeries(stack)
    match.configuration["lambda"] = regularization_lambda

    # 4) Run registration
    try:
        match.run()
    except Exception as e:
        write_log(log_file_path, f"Registration failed: {e}")
        return

    # 5) Retrieve deformed image stack
    deformed = match.get_deformed_images()

    # 6) Save each frame as TIFF
    for i, frame in enumerate(deformed.data):
        norm = (frame - frame.min()) / (frame.max() - frame.min())
        arr = (255 * norm).astype('uint8') if save_dtype == 'uint8' else (65535 * norm).astype('uint16')
        out_tif = os.path.join(output_folder, f"{filename_prefix}{i:03d}.tif")
        try:
            Image.fromarray(arr).save(out_tif)
        except Exception as e:
            write_log(log_file_path, f"Failed to save frame {i}: {e}")

    # 7) Save full aligned stack as HSPY
    stack_out = os.path.join(output_folder, f"{filename_prefix}aligned_stack.hspy")
    try:
        deformed.save(stack_out, overwrite=True)
    except Exception as e:
        write_log(log_file_path, f"Failed to save aligned stack: {e}")

    # 8) Compute stage average and apply denoising
    try:
        avg = deformed.data.mean(axis=0)
        avg_norm = (avg - avg.min()) / (avg.max() - avg.min())

        if denoising_method == 'nlmeans':
            sigma = np.mean(estimate_sigma(avg_norm, channel_axis=None))
            avg_norm = denoise_nl_means(
                avg_norm,
                h=1.15 * sigma,
                fast_mode=True,
                patch_size=5,
                patch_distance=6,
                channel_axis=None
            )
        elif denoising_method.startswith('nlpca'):
            avg_norm = nlpca_denoise(
                avg_norm,
                patch_size=nlpca_patch_size,
                n_clusters=nlpca_n_clusters,
                n_components=nlpca_n_components,
                method=denoising_method
            )

        # Save denoised average as TIFF
        arr_avg = (255 * avg_norm).astype('uint8') if save_dtype == 'uint8' else (65535 * avg_norm).astype('uint16')
        avg_tif = os.path.join(output_folder, f"{filename_prefix}average.tif")
        Image.fromarray(arr_avg).save(avg_tif)

        # Save denoised average as HSPY
        avg_sig = hs.signals.Signal2D(avg_norm.astype('float32'))
        avg_hspy = os.path.join(output_folder, f"{filename_prefix}average.hspy")
        avg_sig.save(avg_hspy, overwrite=True)

        write_log(log_file_path, "Stage average saved (TIFF & HSPY).")
    except Exception as e:
        write_log(log_file_path, f"Stage average failed: {e}")
