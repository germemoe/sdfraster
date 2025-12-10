
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \ 
CUDA_VISIBLE_DEVICES=1 \
uv run train.py  \
--dataset_path /home/jk/data/dataset/DTU/scan37 \
--image_folder images --output_path output/DTU/scan37

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \ 
CUDA_VISIBLE_DEVICES=1 \
uv run train.py  \
--dataset_path /home/jk/data/dataset/DTU/scan24 \
--image_folder images --output_path output/DTU/scan24

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \ 
CUDA_VISIBLE_DEVICES=1 \
uv run train.py  \
--dataset_path /home/jk/data/dataset/DTU/scan40 \
--image_folder images --output_path output/DTU/scan40

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \ 
CUDA_VISIBLE_DEVICES=1 \
uv run train.py  \
--dataset_path /home/jk/data/dataset/DTU/scan55 \
--image_folder images --output_path output/DTU/scan55