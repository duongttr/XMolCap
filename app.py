from inference import Inferencer
from src.utils import set_nested_attr
import yaml
from argparse import Namespace
import gradio as gr
import app_config

# Define two models for molecule captioning
def define_model(model_config_path, ckpt_path):
    args = Namespace()
    model_config = yaml.safe_load(open(model_config_path, 'r'))
    for key, value in model_config.items():
        set_nested_attr(args, key, value)

    args.cuda = False
    args.checkpoint_path = ckpt_path

    model = Inferencer(args)
    return model

lpm24_model = define_model(
    app_config.LPM_24_CFG_PATH,
    app_config.LPM_24_WEIGHTS_PATH
)
chebi20_model = define_model(
    app_config.CHEBI_20_CFG_PATH,
    app_config.CHEBI_20_WEIGHTS_PATH
)

# Define functions for each model

def molecule_captioning(model, molecule_strings, batch_size, num_beams, do_sample, temperature):
    return model.forward(list_of_smiles_seqs=molecule_strings,
                         batch_size=batch_size,
                         num_beams=num_beams,
                         do_sample=do_sample,
                         temperature=temperature)

# Define input components: molecule_strings (list), num_beams, do_sample, temperature
with gr.Blocks() as demo:
    gr.Markdown("# XMolCap: Advancing Molecular Captioning through Multimodal Fusion and Explainable Graph Neural Networks")

    molecule_input = gr.Textbox(label="Molecule Strings (newline-separated)",
                                placeholder="Enter each molecule string on a new line",
                                lines=5)
    num_beams_input = gr.Slider(minimum=1, maximum=10, step=1, label="Number of Beams (Number of hypotheses generated at each step)", value=1)
    do_sample_input = gr.Checkbox(label="Do Sample (Enable sampling for more diverse outputs)", value=False)
    temperature_input = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, label="Temperature (Controls randomness of sampling: higher values mean more random)", value=1.0)
    batch_size_input = gr.Slider(minimum=1, maximum=10, step=1, label="Batch Size (Number of molecules processed in parallel)", value=4)
    
    model_selector = gr.Dropdown(choices=["Trained on L+M-24", "Trained on CheBI-20"], label="Select Model", value="Trained on L+M-24")
    output = gr.Dataframe(headers=["Molecule String", "Generated Caption"], row_count=5, type="array", wrap=True)

    def process_inputs(molecule_input, batch_size, num_beams, do_sample, temperature, model_choice):
        molecule_list = [mol.strip() for mol in molecule_input.split('\n') if mol.strip()]
        if model_choice == "Trained on L+M-24":
            results = molecule_captioning(lpm24_model, molecule_list, batch_size, num_beams, do_sample, temperature)
        else:
            results = molecule_captioning(chebi20_model, molecule_list, batch_size, num_beams, do_sample, temperature)
        return [[mol, cap] for mol, cap in zip(molecule_list, results)]
    
    submit_button = gr.Button("Generate Captions")

    gr.Examples([
        ["COP(=O)(OC)OC(C)=CC(=O)N(C)C\nCC(O)C(=O)N1CCC(C=O)CC1\nCN1C(=O)[N+](c2cncnc2)(C(C)(C)C)CC1C(=O)[O-]", "Trained on L+M-24"],
        ["COC1=CC(=CC(=C1O)OC)C(=O)OC2=CC(=CC(=C2CC(=O)OC)O)O\nC(CC(=O)N)C(C(=O)O)O\nC1=CC(=CC=C1NC(=O)CCN)[N+](=O)[O-]", "Trained on CheBI-20"]
    ],
    inputs=[molecule_input, model_selector],
    outputs=output,
    label="Example Molecule Strings")

    submit_button.click(process_inputs, [molecule_input, batch_size_input, num_beams_input, do_sample_input, temperature_input, model_selector], output)

# Launch the application
demo.launch(share=True)
