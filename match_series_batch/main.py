# main.py

import os
import argparse
import datetime
from tqdm import tqdm
from . import config
from .utils import make_dirs, init_log, write_log
from .processing import process_one_sample

def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch non-rigid image registration processor (with command-line options)")
    parser.add_argument("--input", type=str, help="Input root folder path", default=config.input_root_folder)
    parser.add_argument("--output", type=str, help="Output root folder path", default=config.output_root_folder)
    parser.add_argument("--lambda", type=float, dest="reg_lambda", help="Regularization parameter for deformation", default=config.regularization_lambda)
    parser.add_argument("--prefix", type=str, help="Output filename prefix", default=config.filename_prefix)
    parser.add_argument("--dtype", type=str, choices=['uint8', 'uint16'], help="Save image data type", default=config.save_dtype)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    input_root_folder = args.input
    output_root_folder = args.output
    reg_lambda = args.reg_lambda
    filename_prefix = args.prefix
    save_dtype = args.dtype
    log_file_path = os.path.join(output_root_folder, "processing_log.txt")

    make_dirs(output_root_folder)
    init_log(log_file_path)

    sample_folders = sorted(
        [d for d in os.listdir(input_root_folder) if os.path.isdir(os.path.join(input_root_folder, d))]
    )

    write_log(log_file_path, f"📂 Found {len(sample_folders)} sample folders. Start processing...")

    for sample_name in tqdm(sample_folders, desc="🔄 Processing samples"):
        input_folder = os.path.join(input_root_folder, sample_name)
        output_folder = os.path.join(output_root_folder, sample_name)
        make_dirs(output_folder)

        process_one_sample(
            sample_name, input_folder, output_folder, log_file_path,
            regularization_lambda=reg_lambda,
            filename_prefix=filename_prefix,
            save_dtype=save_dtype
        )

    write_log(log_file_path, f"\n✅ All samples processed. End time: {datetime.datetime.now()}")

if __name__ == "__main__":
    main()
