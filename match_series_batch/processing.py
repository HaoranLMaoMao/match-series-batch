import os
import numpy as np
import hyperspy.api as hs
from pymatchseries import MatchSeries
from PIL import Image
from tqdm import tqdm
from skimage.restoration import denoise_nl_means, estimate_sigma
from .utils import extract_number, write_log

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
        write_log(log_file_path, f"🚀 Starting non-rigid registration for [{sample_name}]...")
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
            dm4_path = os.path.join(output_folder, f"{filename_prefix}{i:03d}.dm4")
            sig.save(dm4_path, overwrite=True)

            img = sig.data
            img_norm = (img - img.min()) / (img.max() - img.min())

            if denoising_method == 'nlmeans':
                sigma_est = np.mean(estimate_sigma(img_norm, channel_axis=None))
                img_norm = denoise_nl_means(
                    img_norm, h=1.15 * sigma_est,
                    fast_mode=True, patch_size=5,
                    patch_distance=6, channel_axis=None
                )
            elif denoising_method == 'nlpca':
                s = hs.signals.Signal2D(img_norm)
                s.decomposition(algorithm='NLPCA', output_dimension=5)
                img_norm = s.get_decomposition_model().data
            elif denoising_method == 'none':
                pass  # No denoising applied

            img_array = (255 * img_norm).astype('uint8') if save_dtype=='uint8' else (65535 * img_norm).astype('uint16')
            tiff_path = os.path.join(output_folder, f"{filename_prefix}{i:03d}.tiff")
            Image.fromarray(img_array).save(tiff_path)
        except Exception as e:
            write_log(log_file_path, f"❌ Failed to save frame {i} for [{sample_name}], Reason: {e}")

    try:
        hspy_path = os.path.join(output_folder, f"{filename_prefix}aligned_stack.hspy")
        deformed.save(hspy_path, overwrite=True)
        write_log(log_file_path, f"📦 Full stack saved: {hspy_path}")
    except Exception as e:
        write_log(log_file_path, f"❌ Failed to save stack for [{sample_name}], Reason: {e}")

    # Save stage average
    try:
        avg_img = deformed.data.mean(axis=0)
        avg_img_norm = (avg_img - avg_img.min()) / (avg_img.max() - avg_img.min())

        if denoising_method == 'nlmeans':
            sigma_est = np.mean(estimate_sigma(avg_img_norm, channel_axis=None))
            avg_img_norm = denoise_nl_means(
                avg_img_norm, h=1.15 * sigma_est,
                fast_mode=True, patch_size=5,
                patch_distance=6, channel_axis=None
            )
        elif denoising_method == 'nlpca':
            s = hs.signals.Signal2D(avg_img_norm)
            s.decomposition(algorithm='NLPCA', output_dimension=5)
            avg_img_norm = s.get_decomposition_model().data
        elif denoising_method == 'none':
            pass

        avg_array = (255 * avg_img_norm).astype('uint8') if save_dtype=='uint8' else (65535 * avg_img_norm).astype('uint16')
        avg_tiff_path = os.path.join(output_folder, f"{filename_prefix}average.tiff")
        Image.fromarray(avg_array).save(avg_tiff_path)

        avg_sig = hs.signals.Signal2D(avg_img_norm)
        avg_dm4_path = os.path.join(output_folder, f"{filename_prefix}average.dm4")
        avg_sig.save(avg_dm4_path, overwrite=True)

        write_log(log_file_path, f"📷 Stage average images saved (TIFF and DM4).")
    except Exception as e:
        write_log(log_file_path, f"❌ Failed to save stage average for [{sample_name}], Reason: {e}")
