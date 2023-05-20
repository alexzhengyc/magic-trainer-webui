import argparse

import gradio as gr

from scripts import presets, ui
from scripts.runner import initialize_runner
from scripts.utilities import args_to_gradio, load_args_template, options_to_gradio



def dict_slice(adict,string):
    keys = adict.keys()
    dict_slice = {}
    if string=="1":
        for k in list(keys)[0:len(list(keys))//2]:
            dict_slice.update({k:adict.get(k)})
    else:
        for k in list(keys)[len(list(keys))//2+1:len(list(keys))]:
            dict_slice.update({k:adict.get(k)})
    return dict_slice

def create_ui():
    network_options = {}

    templates, script_file = load_args_template("Lora.py")

    get_options = lambda: {
        **network_options,
    }
    get_templates = lambda: {
        **templates,
    }

    with gr.Column():
        init_runner = initialize_runner(script_file, get_templates, get_options)
        with gr.Row():
            with gr.Box():
                with gr.Column():   
                    options_to_gradio(dict_slice(templates,"1"), network_options)
            with gr.Box():
                with gr.Column():  
                    options_to_gradio(dict_slice(templates,2), network_options)

    init_runner()
    # init_id()
