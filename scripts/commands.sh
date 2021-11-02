wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nPtsTHPJiuP6wE2bskXFHnYCcVlB_Cwm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nPtsTHPJiuP6wE2bskXFHnYCcVlB_Cwm" -O dataset.zip && rm -rf /tmp/cookies.txt

find . -name ".DS_Store" -delete

https://drive.google.com/file/d/1nPtsTHPJiuP6wE2bskXFHnYCcVlB_Cwm/view?usp=sharing

python3 single-step-learner.py --data_path ../dataset/orbit_benchmark_224 --frame_size 224 \
                         --feature_extractor efficientnetb0 --pretrained_extractor_path features/pretrained/efficientnetb0_imagenet_224.pth \
                         --classifier versa --adapt_features \
                         --context_video_type clean --target_video_type clutter \
                         --with_lite --num_lite_samples 4 --batch_size 4 \
                         --print_by_step \

python3 single-step-learner.py --data_path ../dataset/orbit_benchmark_224 --frame_size 224 \
                         --feature_extractor efficientnetb0  --pretrained_extractor_path features/pretrained/efficientnetb0_imagenet_224.pth \
                         --classifier versa --adapt_features \
                         --context_video_type clean --target_video_type clutter \
                         --with_lite --num_lite_samples 4 --batch_size 4 \
                         --print_by_step \
                         --mode test --model_path checkpoints/orbit_cluve_protonets_efficientnetb0_224_lite.pth --gpu -1 \
