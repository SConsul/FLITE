# Command to download small dataset from google drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nPtsTHPJiuP6wE2bskXFHnYCcVlB_Cwm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nPtsTHPJiuP6wE2bskXFHnYCcVlB_Cwm" -O dataset.zip && rm -rf /tmp/cookies.txt

# Command to delete .DS_Store files created by Macs (causes issues on Linux)
find . -name ".DS_Store" -delete

# Link to Google Drive of small dataset zipped
# https://drive.google.com/file/d/1nPtsTHPJiuP6wE2bskXFHnYCcVlB_Cwm/view?usp=sharing

##### TRAIN commands #####

# Command to train on Protonet LITE efficientnet (on remote)
python3 single-step-learner.py --data_path ../dataset/orbit_benchmark_224 --frame_size 224 \
                         --feature_extractor efficientnetb0 --pretrained_extractor_path features/pretrained/efficientnetb0_imagenet_224.pth \
                         --classifier versa --adapt_features \
                         --context_video_type clean --target_video_type clutter \
                         --with_lite --num_lite_samples 4 --batch_size 4 \
                         --epochs 5 --validation_on_epoch 4 --print_by_step \

# Command to train on Protonet LITE efficientnet (with cpu only on local)
python3 single-step-learner.py --data_path ../dataset/orbit_benchmark_224 --frame_size 224 \
                         --feature_extractor efficientnetb0 --pretrained_extractor_path features/pretrained/efficientnetb0_imagenet_224.pth \
                         --classifier versa --adapt_features \
                         --context_video_type clean --target_video_type clutter \
                         --with_lite --num_lite_samples 4 --batch_size 4 \
                         --epochs 2 --validation_on_epoch 1 --print_by_step \
                         --gpu -1 \

##### TEST commands #####

# Command to test on Protonet LITE efficientnet from checkpoint (on remote)
python3 single-step-learner.py --data_path ../dataset/orbit_benchmark_224 --frame_size 224 \
                         --feature_extractor efficientnetb0  --pretrained_extractor_path features/pretrained/efficientnetb0_imagenet_224.pth \
                         --classifier versa --adapt_features \
                         --context_video_type clean --target_video_type clutter \
                         --with_lite --num_lite_samples 4 --batch_size 4 \
                         --print_by_step \
                         --mode test --from_checkpoint --model_path checkpoint/runorbit_v2/2021-11-02-22-07-56/checkpoint.pt \
                         --clip_length 4

# Command to test on Protonet LITE efficientnet from checkpoint (with cpu only on local)
python3 single-step-learner.py --data_path ../dataset/orbit_benchmark_224 --frame_size 224 \
                         --feature_extractor efficientnetb0  --pretrained_extractor_path features/pretrained/efficientnetb0_imagenet_224.pth \
                         --classifier versa --adapt_features \
                         --context_video_type clean --target_video_type clutter \
                         --with_lite --num_lite_samples 4 --batch_size 4 \
                         --print_by_step \
                         --mode test --from_checkpoint --model_path checkpoint/runorbit_v2/2021-11-02-22-07-56/checkpoint.pt \
                         --gpu -1 --test_tasks_per_user 1 --test_context_num_clips 1
