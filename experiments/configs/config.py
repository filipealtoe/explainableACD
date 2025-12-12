mode = "init_fit"
input_path = "data/raw/tweets_ai.parquet"
output_dir = "experiments/.cache"

chunk_size = 20000
pca_dim = 64
n_clusters = 100
embedding_model = "all-MiniLM-L6-v2"
batch_size_embed = 256
time_freq = "H"
max_init_chunks = None
assign_only = False
