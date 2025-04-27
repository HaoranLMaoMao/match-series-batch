input_root_folder = r"./input"
output_root_folder = r"./output"
log_file_path = output_root_folder + "/processing_log.txt"

regularization_lambda = 20
filename_prefix = "Aligned_"
save_dtype = 'uint8'

# Denoising method: 'nlmeans', 'nlpca', or 'none'
denoising_method = 'nlmeans'

# NLPCA parameters
nlpca_patch_size = 7
nlpca_n_clusters = 10
nlpca_n_components = 8
