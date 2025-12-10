PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \ 
CUDA_VISIBLE_DEVICES=2 \
uv run train.py --eval  \
--dataset_path /home/jk/data/dataset/mipnerf360/room \
--image_folder images_2 --output_path output/room

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \ 
CUDA_VISIBLE_DEVICES=2 \
uv run train.py --eval  \
--dataset_path /home/jk/data/dataset/mipnerf360/flowers \
--image_folder images_4 --output_path output/flowers

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \ 
CUDA_VISIBLE_DEVICES=2 \
uv run train.py --eval  \
--dataset_path /home/jk/data/dataset/mipnerf360/treehill \
--image_folder images_4 --output_path output/treehill