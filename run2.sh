PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \ 
CUDA_VISIBLE_DEVICES=1 \
uv run train.py --eval  \
--dataset_path /home/jk/data/dataset/mipnerf360/kitchen \
--image_folder images_2 --output_path output/kitchen

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \ 
CUDA_VISIBLE_DEVICES=1 \
uv run train.py --eval  \
--dataset_path /home/jk/data/dataset/mipnerf360/counter \
--image_folder images_2 --output_path output/counter

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