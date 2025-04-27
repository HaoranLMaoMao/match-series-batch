import os
import numpy as np
import hyperspy.api as hs
from pymatchseries import MatchSeries
from PIL import Image
from tqdm import tqdm
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import view_as_windows
from sklearn.decomposition import NMF
from .utils import extract_number, write_log

def nl_pca_denoise(image: np.ndarray, patch_size: int = 7, n_clusters: int = 10, n_components: int = 8) -> np.ndarray:
    patches = view_as_windows(image, (patch_size, patch_size))
    patches = patches.reshape(-1, patch_size*patch_size)
    nmf = NMF(n_components=n_components, beta_loss='kullback-leibler', solver='mu', max_iter=300)
    W = nmf.fit_transform(patches)
    H = nmf.components_
    patches_denoised = W @ H
    patches_denoised = patches_denoised.reshape((-1, patch_size, patch_size))
    # Simple averaging to reconstruct
    recon = np.zeros_like(image)
    weight = np.zeros_like(image)
    idx = 0
    for i in range(image.shape[0]-patch_size+1):
        for j in range(image.shape[1]-patch_size+1):
            recon[i:i+patch_size, j:j+patch_size] += patches_denoised[idx]
            weight[i:i+patch_size, j:j+patch_size] += 1
            idx +=1
    return recon/np.maximum(weight,1)

def process_one_sample(sample_name, input_folder, output_folder, log_file_path,
                       regularization_lambda=20, filename_prefix="Aligned_", save_dtype='uint8',
                       denoising_method='nlmeans',
                       nlpca_patch_size=7, nlpca_n_clusters=10, nlpca_n_components=8):
    from pymatchseries import MatchSeries
    os.makedirs(output_folder, exist_ok=True)

    file_list = sorted(
        [f for f in os.listdir(input_folder) if f.endswith(".dm4") and not f.startswith("._")],
        key=extract_number
    )

    if len(file_list) == 0:
        write_log(log_file_path, f"❌ No .dm4 files found in {input_folder}.")
        return

    images = []
    for f in file_list:
        path = os.path.join(input_folder, f)
        try:
            s = hs.load(path, lazy=True)
            images.append(s)
        except Exception as e:
            write_log(log_file_path, f"❌ Failed to load {f}: {e}")

    if len(images) == 0:
        write_log(log_file_path, f"❌ No images loaded in {input_folder}.")
        return

    stack = hs.stack(images)
    match = MatchSeries(stack)
    match.configuration["lambda"] = regularization_lambda

    try:
        match.run()
    except Exception as e:
        write_log(log_file_path, f"❌ MatchSeries run failed: {e}")
        return

    deformed = match.get_deformed_images()

    # Save frames
    for i, img in enumerate(deformed.data):
        img_norm = (img - img.min()) / (img.max() - img.min())
        img_array = (255 * img_norm).astype('uint8') if save_dtype == 'uint8' else (65535 * img_norm).astype('uint16')
        out_path = os.path.join(output_folder, f"{filename_prefix}{i:03d}.tif")
        Image.fromarray(img_array).save(out_path)

    # Save full stack
    stack_out = os.path.join(output_folder, f"{filename_prefix}aligned_stack.hspy")
    deformed.save(stack_out, overwrite=True)

    # Stage average
    try:
        avg_img = deformed.data.mean(axis=0)
        avg_norm = (avg_img - avg_img.min()) / (avg_img.max() - avg_img.min())
        if denoising_method == 'nlmeans':
            sigma_est = np.mean(estimate_sigma(avg_norm, channel_axis=None))
            avg_norm = denoise_nl_means(avg_norm, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6, channel_axis=None)
        elif denoising_method == 'nlpca':
            avg_norm = nl_pca_denoise(avg_norm, patch_size=nlpca_patch_size, n_clusters=nlpca_n_clusters, n_components=nlpca_n_components)

        avg_array = (255 * avg_norm).astype('uint8') if save_dtype == 'uint8' else (65535 * avg_norm).astype('uint16')
        avg_tiff_path = os.path.join(output_folder, f"{filename_prefix}average.tif")
        Image.fromarray(avg_array).save(avg_tiff_path)

        avg_sig = hs.signals.Signal2D(avg_norm.astype('float32'))
        avg_hspy_path = os.path.join(output_folder, f"{filename_prefix}average.hspy")
        avg_sig.save(avg_hspy_path, overwrite=True)

        write_log(log_file_path, f"📷 Stage average images saved (TIFF and HSPY).")
    except Exception as e:
        write_log(log_file_path, f"❌ Failed to save stage average: {e}")

