PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \ 
CUDA_VISIBLE_DEVICES=3 \
uv run train.py --eval  \
--dataset_path /home/jk/data/dataset/mipnerf360/bicycle \
--image_folder images_4 --output_path output/bicycle

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \ 
CUDA_VISIBLE_DEVICES=3 \
uv run train.py --eval  \
--dataset_path /home/jk/data/dataset/mipnerf360/garden \
--image_folder images_4 --output_path output/garden

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \ 
CUDA_VISIBLE_DEVICES=3 \
uv run train.py --eval  \
--dataset_path /home/jk/data/dataset/mipnerf360/stump \
--image_folder images_4 --output_path output/stump

