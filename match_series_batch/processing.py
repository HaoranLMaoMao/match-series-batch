import os
import numpy as np
import datetime
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
    patches = patches.reshape(-1, patch_size * patch_size)
    nmf = NMF(n_components=n_components, beta_loss='kullback-leibler', solver='mu', max_iter=300)
    W = nmf.fit_transform(patches)
    H = nmf.components_
    patches_denoised = W @ H
    patches_denoised = patches_denoised.reshape((-1, patch_size, patch_size))
    recon = np.zeros_like(image)
    weight = np.zeros_like(image)
    idx = 0
    for i in range(image.shape[0] - patch_size + 1):
        for j in range(image.shape[1] - patch_size + 1):
            recon[i:i+patch_size, j:j+patch_size] += patches_denoised[idx]
            weight[i:i+patch_size, j:j+patch_size] += 1
            idx += 1
    return recon / np.maximum(weight, 1)


def find_valid_area(data):
    mask = np.ones_like(data[0], dtype=bool)
    for frame in data:
        mask &= (frame > 0)
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("No valid region found!")
    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0) + 1
    return tuple(slice(top_left[i], bottom_right[i]) for i in range(2))


def process_one_sample(sample_name,
                       input_folder,
                       output_folder,
                       log_file_path,
                       regularization_lambda=20,
                       filename_prefix="Aligned_",
                       save_dtype='uint8',
                       denoising_method='nlmeans',
                       nlpca_patch_size=7,
                       nlpca_n_clusters=10,
                       nlpca_n_components=8):

    os.makedirs(output_folder, exist_ok=True)

    file_list = sorted(
        [f for f in os.listdir(input_folder) if f.endswith('.dm4') and not f.startswith('._')],
        key=extract_number
    )
    write_log(log_file_path, f"\n📁 Processing sample [{sample_name}], {len(file_list)} images found.")

    images = []
    for fname in tqdm(file_list, desc=f"Loading ({sample_name})"):
        path = os.path.join(input_folder, fname)
        try:
            signal = hs.load(path, lazy=True)
            if hasattr(signal, 'data'):
                images.append(signal)
            else:
                write_log(log_file_path, f"⚠️ Non-standard image skipped: {fname}")
        except Exception as e:
            write_log(log_file_path, f"❌ Failed to load {fname}: {e}")
    if not images:
        write_log(log_file_path, f"❌ No valid images for sample [{sample_name}], skipping.")
        return

    try:
        stack = hs.stack(images)
    except Exception as e:
        write_log(log_file_path, f"❌ Failed to stack images for sample [{sample_name}]: {e}")
        return

    try:
        match = MatchSeries(stack)
        match.configuration['lambda'] = regularization_lambda
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        match.path = match.path + f"_{ts}"
        write_log(log_file_path, f"📂 Working directory: {match.path}")
        match.run()
    except Exception as e:
        write_log(log_file_path, f"❌ Registration failed for [{sample_name}], Reason: {e}")
        return

    try:
        deformed = match.get_deformed_images()
    except Exception as e:
        write_log(log_file_path, f"❌ Failed to retrieve deformed images for [{sample_name}]: {e}")
        return

    # Save full aligned stack
    try:
        stack_out = os.path.join(output_folder, f"{filename_prefix}aligned_stack.hspy")
        deformed.save(stack_out, overwrite=True)
        write_log(log_file_path, f"💾 Full aligned stack saved: {stack_out}")
    except Exception as e:
        write_log(log_file_path, f"❌ Failed to save aligned stack for [{sample_name}]: {e}")

    # Save per-frame TIFFs
    for i, img in enumerate(deformed.data):
        frame_path = os.path.join(output_folder, f"{filename_prefix}{i:03d}.tif")
        try:
            arr = np.array(img)
            norm = (arr - arr.min()) / (arr.max() - arr.min())
            img_array = (255 * norm).astype('uint8') if save_dtype == 'uint8' else (65535 * norm).astype('uint16')
            Image.fromarray(img_array).save(frame_path)
        except Exception as e:
            write_log(log_file_path, f"❌ Failed to save frame {i} for [{sample_name}]: {e}")

    # Crop edges and save cropped stack
    try:
        valid_slices = find_valid_area(deformed.data)
        cropped = deformed.isig[valid_slices[0], valid_slices[1]]
        cropped_out = os.path.join(output_folder, f"{filename_prefix}aligned_stack_cropped.hspy")
        cropped.save(cropped_out, overwrite=True)
        write_log(log_file_path, f"✂️ Cropped aligned stack saved: {cropped_out}")
    except Exception as e:
        write_log(log_file_path, f"❌ Cropping failed: {e}")
        return

    # Stage average and denoising on cropped stack
    try:
        avg_img = np.array(cropped.data).mean(axis=0)
        avg_norm = (avg_img - avg_img.min()) / (avg_img.max() - avg_img.min())
        if denoising_method == 'nlmeans':
            sigma = np.mean(estimate_sigma(avg_norm, channel_axis=None))
            avg_norm = denoise_nl_means(avg_norm, h=1.15*sigma, fast_mode=True, patch_size=5, patch_distance=6, channel_axis=None)
        elif denoising_method == 'nlpca':
            avg_norm = nl_pca_denoise(avg_norm, patch_size=nlpca_patch_size, n_clusters=nlpca_n_clusters, n_components=nlpca_n_components)

        avg_array = (255 * avg_norm).astype('uint8') if save_dtype == 'uint8' else (65535 * avg_norm).astype('uint16')
        avg_tiff = os.path.join(output_folder, f"{filename_prefix}average.tif")
        Image.fromarray(avg_array).save(avg_tiff)
        write_log(log_file_path, f"💾 Stage average TIFF saved: {avg_tiff}")

        avg_sig = hs.signals.Signal2D(avg_norm.astype('float32'))
        avg_hspy = os.path.join(output_folder, f"{filename_prefix}average.hspy")
        avg_sig.save(avg_hspy, overwrite=True)
        write_log(log_file_path, f"💾 Stage average HSPY saved: {avg_hspy}")
    except Exception as e:
        write_log(log_file_path, f"❌ Failed to save stage average for [{sample_name}]: {e}")
