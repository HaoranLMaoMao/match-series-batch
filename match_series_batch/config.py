input_root_folder = r"/Users/yourname/Desktop/Data"
output_root_folder = r"/Users/yourname/Desktop/AlignedOutput"
log_file_path = output_root_folder + "/processing_log.txt"

regularization_lambda = 20
filename_prefix = "Aligned_"
save_dtype = 'uint8'

# Default denoising method: 'nlmeans' or 'nlpca'
denoising_method = 'none'
