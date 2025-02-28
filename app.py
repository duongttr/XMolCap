from inference import Inferencer
from src.utils import set_nested_attr
import yaml
from argparse import Namespace
import gradio as gr
import app_config
import os

# Set CUDA_VISIBLE_DEVICES to -1 to disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
    gr.Markdown(app_config.TITLE)
    
    with gr.Row():
        with gr.Column():
            gr.Button(
                app_config.EXTENSIVE_INFO.paper_info.label,
                link=app_config.EXTENSIVE_INFO.paper_info.link,
                size="md"
            ).click(
                lambda _: gr.Info(app_config.EXTENSIVE_INFO.paper_info.action_text)
            )
        with gr.Column():
            gr.Button(
                app_config.EXTENSIVE_INFO.github_info.label,
                link=app_config.EXTENSIVE_INFO.github_info.link,
                size="md"
            ).click(
                lambda _: gr.Info(app_config.EXTENSIVE_INFO.github_info.action_text)
            )
        with gr.Column():
            gr.Button(
                app_config.EXTENSIVE_INFO.webserver_info.label,
                link=app_config.EXTENSIVE_INFO.webserver_info.link,
                size="md"
            ).click(
                lambda _: gr.Info(app_config.EXTENSIVE_INFO.webserver_info.action_text)
            )
    
    gr.Markdown("### Inputs")
    with gr.Row(equal_height=True):
        with gr.Column():
            molecule_input = gr.Textbox(label=app_config.OPTS.molecule_input.label,
                                placeholder=app_config.OPTS.molecule_input.placeholder,
                                lines=app_config.OPTS.molecule_input.lines)
            model_selector = gr.Dropdown(choices=app_config.OPTS.model_selector.choices, 
                                 label=app_config.OPTS.model_selector.label, 
                                 value=app_config.OPTS.model_selector.value)
            gr.Examples(app_config.EXAMPLES.examples,
                        inputs=[molecule_input, model_selector],
                        label=app_config.EXAMPLES.label)
            
        with gr.Column():
            do_sample_input = gr.Checkbox(label=app_config.OPTS.do_sample_input.label, 
                                          value=app_config.OPTS.do_sample_input.value)
            num_beams_input = gr.Slider(minimum=app_config.OPTS.num_beams_input.minimum, 
                                        maximum=app_config.OPTS.num_beams_input.maximum, 
                                        step=app_config.OPTS.num_beams_input.step, 
                                        label=app_config.OPTS.num_beams_input.label, 
                                        value=app_config.OPTS.num_beams_input.value)
            temperature_input = gr.Slider(minimum=app_config.OPTS.temperature_input.minimum, 
                                          maximum=app_config.OPTS.temperature_input.maximum, 
                                          step=app_config.OPTS.temperature_input.step, 
                                          label=app_config.OPTS.temperature_input.label, 
                                          value=app_config.OPTS.temperature_input.value)
            batch_size_input = gr.Slider(minimum=app_config.OPTS.batch_size_input.minimum, 
                                         maximum=app_config.OPTS.batch_size_input.maximum, 
                                         step=app_config.OPTS.batch_size_input.step, 
                                         label=app_config.OPTS.batch_size_input.label, 
                                         value=app_config.OPTS.batch_size_input.value)
            submit_button = gr.Button(app_config.OPTS.submit_button.label, variant=app_config.OPTS.submit_button.variant)
    
    gr.Markdown("### Outputs")
    output = gr.Dataframe(headers=app_config.OPTS.dataframe_output.headers, 
                          row_count=app_config.OPTS.dataframe_output.row_count, 
                          type=app_config.OPTS.dataframe_output.type, 
                          wrap=app_config.OPTS.dataframe_output.wrap)

    def process_inputs(molecule_input, batch_size, num_beams, do_sample, temperature, model_choice):
        molecule_list = [mol.strip() for mol in molecule_input.split('\n') if mol.strip()]
        if len(molecule_list) > app_config.SEQUENCE_LIMIT:
            return gr.Warning(f"Please enter at most {app_config.SEQUENCE_LIMIT} sequences.")
            
        if model_choice == app_config.MODEL_CHOICES[0]:
            results = molecule_captioning(lpm24_model, molecule_list, batch_size, num_beams, do_sample, temperature)
        else:
            results = molecule_captioning(chebi20_model, molecule_list, batch_size, num_beams, do_sample, temperature)
        return [[mol, cap] for mol, cap in zip(molecule_list, results)]

    submit_button.click(process_inputs, [molecule_input, batch_size_input, num_beams_input, do_sample_input, temperature_input, model_selector], output)
    
    # Add a markdown to citation
    gr.Markdown(app_config.CITATION)

# Launch the application
demo.launch(share=True)
