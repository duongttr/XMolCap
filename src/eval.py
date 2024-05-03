from argparse import ArgumentParser, Namespace
from utils import set_nested_attr
import yaml
from lightning_module import T5MultimodalModel
from dataset_module import get_dataloaders
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import csv
import os

def postprocess_text(caption):
    caption = caption[5:]
    if '<eoc>' in caption:
        caption = caption[:caption.index('<eoc>')]
    return caption.strip()

def main(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(args.t5.pretrained_model_name_or_path)
    tokenizer.add_tokens(['α', 'β', 'γ', '<boc>', '<eoc>']) # Add greek symbol, <boc> is start_of_caption, <eoc> is end_of_caption
    
    val_dataloader = get_dataloaders(args, tokenizer, batch_size=args.batch_size, num_workers=4, split='validation')
    
    args.tokenizer = Namespace()
    args.tokenizer.pad_token_id = tokenizer.pad_token_id
    
    model = T5MultimodalModel(args)
    model.resize_token_embeddings(len(tokenizer)) ## Resize due to adding new tokens
    model.to(device)
    
    model.load_state_dict(
        torch.load(args.checkpoint_path, map_location=device)['state_dict'], strict=False
    )
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True) 
    
    with open(args.output_csv, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['selfies', 'gt_caption', 'pred_caption'])
        writer.writeheader()
        
        for idx, batch in enumerate(tqdm(val_dataloader)):
            batch = {k:v.to('cuda') if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            outputs = model.generate_captioning(batch, decoder_start_token_id=35076)
            
            gt_captions = batch['caption']
            pred_captions = [postprocess_text(c) for c in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
            
            writer.writerows([
                {'selfies': selfies,
                 'gt_caption': gt_caption,
                 'pred_caption': pred_caption} for selfies, gt_caption, pred_caption in  zip(batch['selfies'], gt_captions, pred_captions)
            ])
        
        
        

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument('--dataset_name_or_path', type=str, default='ndhieunguyen/LPM-24')
    parser.add_argument('--model_config', type=str, default='./configs/config_use_v_nofg.yaml')
    parser.add_argument('--checkpoint_path', type=str, default='./ckpt/ckpt.pt')
    parser.add_argument('--output_csv', type=str, default='results/output.csv')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--cuda', action='store_true')
    
    args = parser.parse_args()
    
    model_config = yaml.safe_load(open(args.model_config, 'r'))
    for key, value in model_config.items():
        set_nested_attr(args, key, value)
        
    main(args)