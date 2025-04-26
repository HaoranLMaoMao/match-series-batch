import os
import numpy as np
import hyperspy.api as hs
from pymatchseries import MatchSeries
from PIL import Image
from tqdm import tqdm
from skimage.util import view_as_windows
from sklearn.decomposition import NMF
from .utils import extract_number, write_log
import datetime


def nl_pca_denoise(image: np.ndarray, patch_size: int = 7, n_clusters: int = 10, n_components: int = 8) -> np.ndarray:
    """
    Apply Non-Local Poisson PCA denoising as described in Yankovich et al. (2016).
    - image: 2D numpy array (normalized to [0,1])
    - patch_size: size of square patch
    - n_clusters: number of patch clusters
    - n_components: number of PCA components per cluster
    Returns denoised image of same shape.
    """
    M, N = image.shape
    L = 1  # single-channel denoising

    # 1. Extract overlapping patches
    patches = view_as_windows(image, (patch_size, patch_size), step=1)
    num_patches = patches.shape[0] * patches.shape[1]
    X = patches.reshape(num_patches, patch_size * patch_size)

    # 2. Simple k-means on patch sum to cluster (approximate Poisson clustering)
    # Here we use patch total count as feature
    totals = X.sum(axis=1).reshape(-1, 1)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(totals)

    # 3. Per-cluster Poisson PCA via NMF (KL divergence)
    X_denoised = np.zeros_like(X)
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            continue
        Xk = X[idx]
        model = NMF(n_components=n_components,
                    init='nndsvda', solver='mu', beta_loss='kullback-leibler', max_iter=300)
        W = model.fit_transform(Xk)
        H = model.components_
        Xk_hat = np.dot(W, H)
        X_denoised[idx] = Xk_hat

    # 4. Reconstruct image by averaging overlapping patches
    denoised = np.zeros_like(image)
    weight = np.zeros_like(image)
    idx = 0
    for i in range(M - patch_size + 1):
        for j in range(N - patch_size + 1):
            patch = X_denoised[idx].reshape(patch_size, patch_size)
            denoised[i:i+patch_size, j:j+patch_size] += patch
            weight[i:i+patch_size, j:j+patch_size] += 1
            idx += 1
    denoised /= np.maximum(weight, 1)
    return denoised


def process_one_sample(sample_name, input_folder, output_folder, log_file_path,
                       regularization_lambda=20, filename_prefix="Aligned_", save_dtype='uint8',
                       denoising_method='nlmeans'):
    file_list = sorted(
        [f for f in os.listdir(input_folder) if f.endswith(".dm4") and not f.startswith("._")],
        key=extract_number
    )
    if not file_list:
        write_log(log_file_path, f"⚠️ Sample [{sample_name}] is empty, skipping.")
        return

    write_log(log_file_path, f"\n📁 Processing sample [{sample_name}], {len(file_list)} images found.")
    images = []
    for f in tqdm(file_list, desc=f"📥 Loading ({sample_name})"):
        path = os.path.join(input_folder, f)
        try:
            signal = hs.load(path, lazy=True)
            if hasattr(signal, "data"):
                images.append(signal)
            else:
                write_log(log_file_path, f"⚠️ Non-standard image (skipped): {f}")
        except Exception as e:
            write_log(log_file_path, f"❌ Failed to load (skipped): {f}, Reason: {e}")

    if not images:
        write_log(log_file_path, f"❌ No valid images for sample [{sample_name}], skipping.")
        return

    try:
        stack = hs.stack(images)
    except Exception as e:
        write_log(log_file_path, f"❌ Failed to stack images for sample [{sample_name}], Reason: {e}")
        return

    try:
        match = MatchSeries(stack)
        match.configuration["lambda"] = regularization_lambda
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        match.path = match.path + f"_{now}"
        write_log(log_file_path, f"📂 Generated new working directory: {match.path}")
        match.run()
    except Exception as e:
        write_log(log_file_path, f"❌ Registration failed for [{sample_name}], Reason: {e}")
        return

    try:
        deformed = match.get_deformed_images()
    except Exception as e:
        write_log(log_file_path, f"❌ Failed to retrieve deformed images for [{sample_name}], Reason: {e}")
        return

    write_log(log_file_path, f"💾 Saving frames for sample [{sample_name}]...")
    for i, sig in enumerate(tqdm(deformed, desc=f"💾 Saving frames ({sample_name})")):
        try:
            img = sig.data
            img_norm = (img - img.min()) / (img.max() - img.min())
            # Apply denoising per frame
            if denoising_method == 'nlmeans':
                from skimage.restoration import denoise_nl_means, estimate_sigma
                sigma_est = np.mean(estimate_sigma(img_norm, channel_axis=None))
                img_norm = denoise_nl_means(img_norm, h=1.15 * sigma_est,
                                             fast_mode=True, patch_size=5,
                                             patch_distance=6, channel_axis=None)
            elif denoising_method == 'nlpca':
                img_norm = nl_pca_denoise(img_norm, patch_size=7, n_clusters=10, n_components=8)
            # 'none' leaves img_norm unchanged

            img_array = (255 * img_norm).astype('uint8') if save_dtype=='uint8' else (65535 * img_norm).astype('uint16')
            tiff_path = os.path.join(output_folder, f"{filename_prefix}{i:03d}.tif")
            Image.fromarray(img_array).save(tiff_path)
        except Exception as e:
            write_log(log_file_path, f"❌ Failed to save frame {i} for [{sample_name}], Reason: {e}")

    try:
        hspy_path = os.path.join(output_folder, f"{filename_prefix}aligned_stack.hspy")
        deformed.save(hspy_path, overwrite=True)
        write_log(log_file_path, f"📦 Full stack saved: {hspy_path}")
    except Exception as e:
        write_log(log_file_path, f"❌ Failed to save stack for [{sample_name}], Reason: {e}")

    try:
        avg_img = deformed.data.mean(axis=0)
        avg_norm = (avg_img - avg_img.min()) / (avg_img.max() - avg_img.min())
        if denoising_method == 'nlmeans':
            from skimage.restoration import denoise_nl_means, estimate_sigma
            sigma_est = np.mean(estimate_sigma(avg_norm, channel_axis=None))
            avg_norm = denoise_nl_means(avg_norm, h=1.15 * sigma_est,
                                         fast_mode=True, patch_size=5,
                                         patch_distance=6, channel_axis=None)
        elif denoising_method == 'nlpca':
            avg_norm = nl_pca_denoise(avg_norm, patch_size=7, n_clusters=10, n_components=8)

        avg_array = (255 * avg_norm).astype('uint8') if save_dtype=='uint8' else (65535 * avg_norm).astype('uint16')
        avg_tiff_path = os.path.join(output_folder, f"{filename_prefix}average.tif")
        Image.fromarray(avg_array).save(avg_tiff_path)

        avg_sig = hs.signals.Signal2D(np.array(avg_norm))
        avg_hspy_path = os.path.join(output_folder, f"{filename_prefix}average.hspy")
        avg_sig.save(avg_hspy_path, overwrite=True)

        write_log(log_file_path, f"📷 Stage average images saved (TIFF and HSPY).")
    except Exception as e:
        write_log(log_file_path, f"❌ Failed to save stage average for [{sample_name}], Reason: {e}")
