import os
from types import SimpleNamespace

LPM_24_CFG_PATH = os.path.join(
    os.path.dirname(__file__), 'src/configs/config_lpm24_train.yaml'
)
LPM_24_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), 'checkpoints/lpm24_center_layer_smiles_off.ckpt'
)
CHEBI_20_CFG_PATH = os.path.join(
    os.path.dirname(__file__), 'src/configs/config_chebi20_train.yaml'
)
CHEBI_20_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), 'checkpoints/chebi20_center_layer_all_modals.ckpt'
)

TITLE = "# XMolCap: Advancing Molecular Captioning through Multimodal Fusion and Explainable Graph Neural Networks"

EXTENSIVE_INFO = SimpleNamespace(
    paper_info=SimpleNamespace(
        label="📄 Paper",
        link="",
        action_text="Paper link will be updated soon."
    ),
    github_info=SimpleNamespace(
        label="🐙 GitHub",
        link="https://github.com/cbbl-skku-org/XMolCap",
        action_text="You are now being redirected to the GitHub repository."
    ),
    webserver_info=SimpleNamespace(
        label="🌐 Webserver (CBBL@SKKU)",
        link="",
        action_text="Webserver link will be updated soon."
    )
)

MODEL_CHOICES = ["Trained on L+M-24", "Trained on CheBI-20"]

SEQUENCE_LIMIT = 10

OPTS = SimpleNamespace(
    molecule_input=SimpleNamespace(
        label=f"⚛️ Molecule Strings (newline-separated, Limit: {SEQUENCE_LIMIT} sequences)",
        placeholder="Enter each molecule string on a new line",
        lines=5
    ),
    num_beams_input=SimpleNamespace(
        minimum=1,
        maximum=5,
        step=1,
        label="☄️ Number of Beams (Number of hypotheses generated at each step)",
        value=1   
    ),
    do_sample_input=SimpleNamespace(
        label="🧪 Do Sample (Enable sampling for more diverse outputs)",
        value=False
    ),
    temperature_input=SimpleNamespace(
        minimum=0.1,
        maximum=2.0,
        step=0.1,
        label="🌡️ Temperature (Controls randomness of sampling: higher values mean more random)",
        value=1.0
    ),
    batch_size_input=SimpleNamespace(
        minimum=1,
        maximum=10,
        step=1,
        label="📦 Batch Size (Number of molecules processed in parallel)",
        value=4
    ),
    model_selector=SimpleNamespace(
        choices=MODEL_CHOICES,
        label="🗂️ Select Model",
        value=MODEL_CHOICES[0]
    ),
    dataframe_output=SimpleNamespace(
        headers=["Molecule String", "Generated Caption"],
        row_count=5,
        type="array",
        wrap=True
    ),
    submit_button=SimpleNamespace(
        label="✍ Generate Captions",
        variant="primary"
    )
)

EXAMPLES = SimpleNamespace(
    label="Example Molecule Strings",
    examples=[
        ["COP(=O)(OC)OC(C)=CC(=O)N(C)C\nCC(O)C(=O)N1CCC(C=O)CC1\nCN1C(=O)[N+](c2cncnc2)(C(C)(C)C)CC1C(=O)[O-]", MODEL_CHOICES[0]],
        ["COC1=CC(=CC(=C1O)OC)C(=O)OC2=CC(=CC(=C2CC(=O)OC)O)O\nC(CC(=O)N)C(C(=O)O)O\nC1=CC(=CC=C1NC(=O)CCN)[N+](=O)[O-]", MODEL_CHOICES[1]]
    ]
)

CITATION = """
### Citation
If you are interested in this work, please cite the following paper:
```
@ARTICLE{11012653,
  author={Tran, Duong Thanh and Nguyen, Nguyen Doan Hieu and Pham, Nhat Truong and Rakkiyappan, Rajan and Karki, Rajendra and Manavalan, Balachandran},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={XMolCap: Advancing Molecular Captioning through Multimodal Fusion and Explainable Graph Neural Networks}, 
  year={2025},
  volume={},
  number={},
  pages={1-12},
  keywords={Biological system modeling;Feature extraction;Chemicals;Bioinformatics;Accuracy;Training;Data models;Data mining;Transformers;Encoding;Explainable artificial intelligence;graph neural networks;language and molecules;large language models;molecular captioning;model interpretation;multimodal fusion},
  doi={10.1109/JBHI.2025.3572910}}
```
"""