from src.lightning_module import T5MultimodalModel
from transformers import AutoTokenizer
import torch
from torchvision import transforms
from src.backbones.graph.graph_featurizer import GraphFeaturizer
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Batch

class Inferencer:
    def __init__(self, args):
        device = torch.device('cuda' if args.cuda else 'cpu')
        
        # Initialize preprocessor
        ## Language model
        self.lm_tokenizer = AutoTokenizer.from_pretrained(args.t5.pretrained_model_name_or_path)
        self.lm_tokenizer.add_tokens(['α', 'β', 'γ', '<boc>', '<eoc>'])
        
        ## Image
        
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        ## SMILES
        self.smi_tokenizer = AutoTokenizer.from_pretrained(args.roberta.pretrained_model_name_or_path)
        
        ## Graph
        self.graph_featurizer = GraphFeaturizer({'name' : 'ogb'})
        
        # Initialize model
        self.model = T5MultimodalModel(args)
        self.model.resize_token_embeddings(len(self.lm_tokenizer))
        self.model.to(device)
        self.model.eval()
        self.model.tokenizer = self.lm_tokenizer
        self.model.load_state_dict(
            torch.load(args.checkpoint_path, map_location=device)['state_dict'], strict=False
        )
        
        # Initialize prompt
        self.task_definition = 'Definition: You are given a molecule SELFIES. Your job is to generate the molecule description in English that fits the molecule SELFIES.\n\n'
        self.task_input = 'Now complete the following example -\nInput: <bom>{selfies_str}<eom>\nOutput: '
        self.model_input = self.task_definition + self.task_input
    
    def _draw_molecule(self, smiles_seq):
        mol = Chem.MolFromSmiles(smiles_seq)
        img = Draw.MolToImage(mol)
        return img
    
    def _batch(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    def _prepare_inputs_batch(self, batch_smiles_seqs, batch_size=4):
        
        lm_inputs_ids_batch = []
        lm_attention_masks_batch = []
        smi_inputs_ids_batch = []
        smi_attention_masks_batch = []
        images_batch = []
        graphs_batch = []
        labels_batch = []
        
        for smiles_seq in batch_smiles_seqs:
            # Normalize SMILES to canonical form
            smiles_seq = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_seq))
            
            # convert SMILES to SELFIES
            selfies_seq = sf.encoder(smiles_seq)
            
            lm_input = self.lm_tokenizer(
                self.model_input.format(selfies_str=selfies_seq),
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            smi_input = self.smi_tokenizer(
                smiles_seq,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            img_input = self.image_transform(self._draw_molecule(smiles_seq))
            
            lm_input_ids = lm_input['input_ids'].flatten()
            lm_attention_mask = lm_input['attention_mask'].flatten()
            
            smi_input_ids = smi_input['input_ids'].flatten()
            smi_attention_mask = smi_input['attention_mask'].flatten()
            
            graph = self.graph_featurizer(smiles_seq)
            
            # Empty labels
            labels = torch.tensor([])
            
            lm_inputs_ids_batch.append(lm_input_ids)
            lm_attention_masks_batch.append(lm_attention_mask)
            smi_inputs_ids_batch.append(smi_input_ids)
            smi_attention_masks_batch.append(smi_attention_mask)
            images_batch.append(img_input)
            graphs_batch.append(graph)
            labels_batch.append(labels)
        
        data_batch = {
            'input_ids' : torch.stack(lm_inputs_ids_batch),
            'attention_mask' : torch.stack(lm_attention_masks_batch),
            'smiles_input_ids' : torch.stack(smi_inputs_ids_batch),
            'smiles_attention_mask' : torch.stack(smi_attention_masks_batch),
            'images' : torch.stack(images_batch),
            'graph' : Batch.from_data_list(graphs_batch),
            'labels' : torch.stack(labels_batch)
        }
        
        return data_batch
    
    def forward(self, list_of_smiles_seqs, 
                batch_size=4,
                num_beams=1,
                do_sample=False,
                temperature=1.0):
        ALL_OUTPUTS = []
        for smiles_batch in self._batch(list_of_smiles_seqs, batch_size):
            inputs_batch = self._prepare_inputs_batch(smiles_batch)
            outputs = self.model.generate_captioning(
                inputs_batch,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature
            )
            ALL_OUTPUTS.extend(outputs)
        return ALL_OUTPUTS