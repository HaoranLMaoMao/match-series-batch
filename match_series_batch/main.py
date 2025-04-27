import argparse, os
from tqdm import tqdm
from . import config
from .utils import make_dirs, init_log, write_log
from .processing import process_one_sample

def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch non-rigid image registration processor with flexible denoising and NLPCA options.")
    parser.add_argument("--input", type=str, default=config.input_root_folder, help="Input root folder")
    parser.add_argument("--output", type=str, default=config.output_root_folder, help="Output root folder")
    parser.add_argument("--lambda", type=float, dest="reg_lambda", default=config.regularization_lambda, help="Deformation regularization parameter")
    parser.add_argument("--prefix", type=str, default=config.filename_prefix, help="Output filename prefix")
    parser.add_argument("--dtype", type=str, choices=["uint8","uint16"], default=config.save_dtype, help="Output data type")
    parser.add_argument("--denoising", type=str, choices=["nlmeans","nlpca","none"], default=config.denoising_method, help="Denoising method")
    parser.add_argument("--nlpca_patch_size", type=int, default=config.nlpca_patch_size, help="NLPCA patch size")
    parser.add_argument("--nlpca_n_clusters", type=int, default=config.nlpca_n_clusters, help="NLPCA number of clusters")
    parser.add_argument("--nlpca_n_components", type=int, default=config.nlpca_n_components, help="NLPCA number of components")
    return parser.parse_args()

def main():
    args = parse_arguments()
    input_root, output_root = args.input, args.output
    make_dirs(output_root)
    init_log(os.path.join(output_root, "processing_log.txt"))
    folders = sorted([d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))])
    write_log(os.path.join(output_root, "processing_log.txt"), f"Found {len(folders)} samples.")
    for sample in tqdm(folders, desc="Processing"):
        in_f, out_f = os.path.join(input_root, sample), os.path.join(output_root, sample)
        make_dirs(out_f)
        process_one_sample(sample, in_f, out_f, os.path.join(output_root, "processing_log.txt"),
                           regularization_lambda=args.reg_lambda,
                           filename_prefix=args.prefix, save_dtype=args.dtype,
                           denoising_method=args.denoising,
                           nlpca_patch_size=args.nlpca_patch_size,
                           nlpca_n_clusters=args.nlpca_n_clusters,
                           nlpca_n_components=args.nlpca_n_components)
    write_log(os.path.join(output_root, "processing_log.txt"), "All done.")
