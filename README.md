# mol-lang-bridge
Language and Molecules Translation for Scientific Insight and Discovery

## Fine-tuning baseline BioT5 on LPM-24 dataset
1. Setup environments
```zsh
conda create -n ACLMol python=3.9
conda activate ACLMol
python -m pip install -r requirements.txt
```

2. Login WandB and HuggingFace
```zsh
python login.py --hf_token <hf_token> --wandb_key <wandb_key>
```

3. Train model
```zsh
python biot5_base/main.py --epochs 100 --batch_size 8 --num_workers 4 --cuda --precision '32'
```

## Pretraining Swin MIM
1. Download data
[Download data](https://www.kaggle.com/datasets/duongtran1909/zinc20-1m) from Kaggle

2. Train model
```zsh
python src/mim_pretraining.py --model_config src/config/swin_config.yaml \
                              --train_data_path path/to/train_data_folder \
                              --valid_data_path path/to/valid_data_folder \
                              --epochs 100 \
                              --batch_size 196 \
                              --num_workers 8  \
                              --cuda \
                              --precision '32' \
                              --num_devices 1
```
