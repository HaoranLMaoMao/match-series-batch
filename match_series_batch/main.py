import argparse
import os
from tqdm import tqdm
from . import config
from .utils import make_dirs, init_log, write_log
from .processing import process_one_sample

def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch non-rigid image registration processor with optional denoising")
    parser.add_argument("--input", type=str, default=config.input_root_folder, help="Input root folder")
    parser.add_argument("--output", type=str, default=config.output_root_folder, help="Output root folder")
    parser.add_argument("--lambda", type=float, dest="reg_lambda", default=config.regularization_lambda, help="Deformation regularization")
    parser.add_argument("--prefix", type=str, default=config.filename_prefix, help="Output filename prefix")
    parser.add_argument("--dtype", type=str, choices=["uint8", "uint16"], default=config.save_dtype, help="Output data type")
    parser.add_argument("--denoising", type=str, choices=["nlmeans", "nlpca", "none"], default=config.denoising_method, help="Denoising method")
    return parser.parse_args()

def main():
    args = parse_arguments()

    input_root_folder = args.input
    output_root_folder = args.output
    reg_lambda = args.reg_lambda
    filename_prefix = args.prefix
    save_dtype = args.dtype
    denoising_method = args.denoising
    log_file_path = os.path.join(output_root_folder, "processing_log.txt")

    make_dirs(output_root_folder)
    init_log(log_file_path)

    sample_folders = sorted(
        [d for d in os.listdir(input_root_folder) if os.path.isdir(os.path.join(input_root_folder, d))]
    )

    write_log(log_file_path, f"📂 Found {len(sample_folders)} samples. Start processing...")

    for sample_name in tqdm(sample_folders, desc="🔄 Processing samples"):
        input_folder = os.path.join(input_root_folder, sample_name)
        output_folder = os.path.join(output_root_folder, sample_name)
        make_dirs(output_folder)

        process_one_sample(
            sample_name, input_folder, output_folder, log_file_path,
            regularization_lambda=reg_lambda,
            filename_prefix=filename_prefix,
            save_dtype=save_dtype,
            denoising_method=denoising_method
        )

    write_log(log_file_path, f"✅ All samples processed.")
