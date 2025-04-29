import os
import datetime
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

def extract_patches(image, patch_size):
    patches = view_as_windows(image, (patch_size, patch_size))
    return patches.reshape(-1, patch_size * patch_size)

def reconstruct_from_patches(patches_denoised, image_shape, patch_size, idxs):
    recon = np.zeros(image_shape, dtype=float)
    weight = np.zeros(image_shape, dtype=float)
    coords = np.argwhere(np.ones((image_shape[0] - patch_size + 1,
                                  image_shape[1] - patch_size + 1)))
    for k, idx in enumerate(idxs):
        i, j = coords[idx]
        recon[i:i+patch_size, j:j+patch_size] += patches_denoised[k]
        weight[i:i+patch_size, j:j+patch_size] += 1
    return recon / np.maximum(weight, 1)

def nlpca_denoise(image, patch_size, n_clusters, n_components, method):
    patches = extract_patches(image, patch_size)
    # Clustering
    if method == 'nlpca-kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        labels = clusterer.fit_predict(patches)
    elif method == 'nlpca-spectral':
        clusterer = SpectralClustering(
            n_clusters=n_clusters, assign_labels='kmeans', affinity='nearest_neighbors',
            n_neighbors=10, random_state=0, n_jobs=-1
        )
        labels = clusterer.fit_predict(patches)
    elif method == 'nlpca-gmm':
        clusterer = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0)
        labels = clusterer.fit_predict(patches)
    else:
        raise ValueError(f"Unsupported NLPCA method: {method}")
    # NMF per cluster
    recon = np.zeros(image.shape, dtype=float)
    weight = np.zeros(image.shape, dtype=float)
    for c in range(n_clusters):
        idxs = np.where(labels == c)[0]
        if idxs.size == 0:
            continue
        group = patches[idxs]
        nmf = NMF(n_components=n_components, beta_loss='kullback-leibler', solver='mu', max_iter=300)
        W = nmf.fit_transform(group)
        H = nmf.components_
        denoised_patches = (W @ H).reshape(-1, patch_size, patch_size)
        partial = reconstruct_from_patches(denoised_patches, image.shape, patch_size, idxs)
        recon += partial
        weight += (weight == 0)
    return recon / np.maximum(weight, 1)

def process_one_sample(sample_name, input_folder, output_folder, log_file_path,
                       regularization_lambda=20, filename_prefix="Aligned_",
                       save_dtype='uint8', denoising_method='nlmeans',
                       nlpca_patch_size=7, nlpca_n_clusters=10, nlpca_n_components=8):
    """
    Pipeline for one sample:
      1. Non-rigid registration via pyMatchSeries
      2. Save each frame TIFF + full aligned stack HSPY
      3. Compute stage average
      4. Apply denoising: none, nlmeans, nlpca-kmeans, nlpca-spectral, nlpca-gmm
      5. Save denoised average TIFF + HSPY
    """
    from pymatchseries import MatchSeries

    os.makedirs(output_folder, exist_ok=True)

    # Gather input files
    files = sorted(f for f in os.listdir(input_folder)
                   if f.endswith('.dm4') and not f.startswith('._'))
    if not files:
        write_log(log_file_path, f"❌ No .dm4 files in {input_folder}")
        return

    # Load images
    images = []
    for f in files:
        path = os.path.join(input_folder, f)
        try:
            sig = hs.load(path, lazy=True)
            images.append(sig)
        except Exception as e:
            write_log(log_file_path, f"❌ Load failed {f}: {e}")
    if not images:
        write_log(log_file_path, f"❌ No valid images in {input_folder}")
        return

    # Stack & configure
    stack = hs.stack(images)
    match = MatchSeries(stack)
    match.configuration['lambda'] = regularization_lambda
    # Create unique directory to avoid overwrite prompt
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    match.path = f"{match.path}_{now}"
    write_log(log_file_path, f"📂 New working dir: {match.path}")

    # Run registration
    try:
        match.run()
    except Exception as e:
        write_log(log_file_path, f"❌ Registration failed for [{sample_name}]: {e}")
        return

    deformed = match.get_deformed_images()

    # Save frames
    for i, frame in enumerate(deformed.data):
        norm = (frame - frame.min()) / (frame.max() - frame.min())
        arr = (255*norm).astype('uint8') if save_dtype=='uint8' else (65535*norm).astype('uint16')
        out_tif = os.path.join(output_folder, f"{filename_prefix}{i:03d}.tif")
        try:
            Image.fromarray(arr).save(out_tif)
        except Exception as e:
            write_log(log_file_path, f"❌ Save frame {i} failed: {e}")

    # Save full stack
    stack_out = os.path.join(output_folder, f"{filename_prefix}aligned_stack.hspy")
    try:
        deformed.save(stack_out, overwrite=True)
    except Exception as e:
        write_log(log_file_path, f"❌ Save stack failed: {e}")

    # Stage average + denoise
    try:
        avg = deformed.data.mean(axis=0)
        norm_avg = (avg - avg.min()) / (avg.max() - avg.min())
        if denoising_method == 'nlmeans':
            sigma = np.mean(estimate_sigma(norm_avg, channel_axis=None))
            norm_avg = denoise_nl_means(norm_avg, h=1.15*sigma,
                                        fast_mode=True, patch_size=5, patch_distance=6,
                                        channel_axis=None)
        elif denoising_method.startswith('nlpca'):
            norm_avg = nlpca_denoise(norm_avg,
                                     patch_size=nlpca_patch_size,
                                     n_clusters=nlpca_n_clusters,
                                     n_components=nlpca_n_components,
                                     method=denoising_method)
        arr_avg = (255*norm_avg).astype('uint8') if save_dtype=='uint8' else (65535*norm_avg).astype('uint16')
        # Save average TIFF
        avg_tif = os.path.join(output_folder, f"{filename_prefix}average.tif")
        Image.fromarray(arr_avg).save(avg_tif)
        # Save average HSPY
        avg_sig = hs.signals.Signal2D(norm_avg.astype('float32'))
        avg_hspy = os.path.join(output_folder, f"{filename_prefix}average.hspy")
        avg_sig.save(avg_hspy, overwrite=True)
        write_log(log_file_path, "📷 Stage average saved.")
    except Exception as e:
        write_log(log_file_path, f"❌ Stage average failed: {e}")

