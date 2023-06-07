import ast
import importlib
import os
import subprocess
import sys

import gradio as gr

import scripts.shared as shared
from scripts.shared import ROOT_DIR,SD_DIR
MODEL_DIR=os.path.join(SD_DIR,"models")
python = sys.executable


def path_to_module(filepath):
    return (
        os.path.relpath(filepath, ROOT_DIR).replace(os.path.sep, ".").replace(".py", "")
    )


def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def literal_eval(v, module=None):
    if v == "str":
        return str
    elif v == "int":
        return int
    elif v == "float":
        return float
    elif v == list:
        return list
    else:
        if module:
            try:
                m = importlib.import_module(module)
                if hasattr(m, v):
                    return getattr(m, v)
            except:
                ()

        return ast.literal_eval(v)


def compile_arg_parser(txt, module_path=None):
    in_parser = False
    parsers = {}
    args = []
    arg = ""
    in_list = False
    in_str = None

    def compile(arg):
        arg = arg.strip()
        matches = arg.split("=")

        if len(matches) > 1:
            k = "".join(matches[:1])
            v = literal_eval("".join(matches[1:]), module_path)
            return (k, v)
        else:
            return literal_eval(arg, module_path)

    for line in txt.split("\n"):
        line = line.split("#")[0]

        if "parser.add_argument(" in line:
            in_parser = True
            line = line.replace("parser.add_argument(", "")

        if not in_parser:
            continue

        for c in line:

            if in_str is None and c == ")":
                if arg.strip():
                    args.append(compile(arg))
                in_parser = False
                [dest, *others] = args
                parsers[dest] = {"dest": dest.replace("--", ""), **dict(others)}
                arg = ""
                args = []
                break

            if c == "[":
                in_list = True
            elif c == "]":
                in_list = False
            if c == '"' or c == "'":
                if in_str is not None and in_str == c:
                    in_str = None
                elif in_str is None:
                    in_str = c

            if c == "," and not in_list and in_str is None:
                args.append(compile(arg))
                arg = ""
                continue

            arg += c

    if arg.strip():
        args.append(compile(arg))
    return parsers


def load_args_template(*filename):
    repo_dir = os.path.join(ROOT_DIR, "kohya_ss_revised")
    filepath = os.path.join(repo_dir, *filename)
    with open(filepath, mode="r", encoding="utf-8_sig") as f:
        lines = f.readlines()
        add = False
        txt = ""
        for line in lines:
            if add == True:
                txt += line
            if "def setup_parser()" in line:
                add = True
                continue
    return compile_arg_parser(txt, path_to_module(filepath)), filepath


def check_key(d, k):
    return k in d and d[k] is not None


def get_arg_type(d):
    if check_key(d, "choices"):
        return list
    if check_key(d, "type"):
        return d["type"]
    if check_key(d, "action") and (
        d["action"] == "store_true" or d["action"] == "store_false"
    ):
        return bool
    if check_key(d, "const") and type(d["const"]) == bool:
        return bool
    return str


def options_to_gradio(options, out, overrides={}):
    scripts_dir = os.path.dirname(os.path.realpath(__file__))
    magic_trainer_dir = os.path.dirname(scripts_dir)
    extensions_dir = os.path.dirname(magic_trainer_dir)
    stable_diffusion_dir = os.path.dirname(extensions_dir)
    models_dir = os.path.join(stable_diffusion_dir, "models")

    for _, item in options.items():
        item = item.__dict__ if hasattr(item, "__dict__") else item
        key = item["dest"]
        if key == "help":
            continue
        override = overrides[key] if key in overrides else {}
        component = None

        help = item["help"] if "help" in item else ""
        id = f"magic_trainer_webui__{shared.current_tab.replace('.', '_')}_{key}"
        type = override["type"] if "type" in override else get_arg_type(item)
        
        if key=="sd_model":
            choices=[]
            # datanames = os.listdir(os.path.join(MODEL_DIR, "Stable-diffusion"))
            datanames = os.listdir("/root/stable-diffusion-webui/models/Stable-diffusion")
            for dataname in datanames:
                # if os.path.splitext(dataname)[1] == '.ckpt' or os.path.splitext(dataname)[1] == '.safetensors':
                choices.append(os.path.join(models_dir, "Stable-diffusion", dataname))
            if len(choices)==0:
                choices.append("")
            component = gr.Dropdown(
                choices=choices,
                value=item["default"] if check_key(item, "default") else choices[0],
                label="sd_model (read from stable-diffusion/models/Stable-diffusion)",
                elem_id=id,
                interactive=True,
            )

        elif key=="vae":
            choices=[]
            datanames = os.listdir(os.path.join(MODEL_DIR, "VAE"))
            for dataname in datanames:
                if os.path.splitext(dataname)[1] == '.ckpt' or os.path.splitext(dataname)[1] == '.safetensors':
                    choices.append(os.path.join(models_dir, "VAE", dataname))
            choices.append("None")
            component = gr.Dropdown(
                choices=choices,
                value=item["default"] if check_key(item, "default") else "None",
                label="vae (read from stable-diffusion/models/VAE)",
                elem_id=id,
                interactive=True,
            )

        elif key=="blip":
            choices=[]
            datanames = os.listdir(os.path.join(models_dir, "BLIP"))
            for dataname in datanames:
                choices.append(os.path.join(models_dir, "BLIP", dataname))
            choices.append("download from huggingface")

            # choices.append("")
            component = gr.Dropdown(
                choices=choices,
                value=item["default"] if check_key(item, "default") else choices[0],
                label="blip: blip model (read from stable-diffusion/models/BLIP)",
                elem_id=id,
                interactive=True,
            )

        elif key== "train_data":
            component = gr.Textbox(
                value=item["default"] if check_key(item, "default") else "",
                label="train/instance dataset (/root/dir/to/folder)",
                elem_id=id,
                interactive=True,
            ).style()

        elif key== "reg_data":
            component = gr.Textbox(
                value=item["default"] if check_key(item, "default") else "",
                label="reg/class dataset (/root/dir/to/folder)",
                elem_id=id,
                interactive=True,
            ).style()

        elif key == "train_data_dir":
            component = gr.Textbox(
                value=item["default"] if check_key(item, "default") else "",
                label="image folder (/root/dir/to/folder)",
                elem_id=id,
                interactive=True,
            ).style()

        elif key== "extra_sd_path":
            component = gr.Textbox(
                value=item["default"] if check_key(item, "default") else "",
                label="extra sd path (use this if not empty)",
                elem_id=id,
                interactive=True,
            ).style()

        elif key== "instance_token":
            component = gr.Textbox(
                value=item["default"] if check_key(item, "default") else "",
                label="instance token (use if no text prompts in train dataset)",
                elem_id=id,
                interactive=True,
            ).style()   
        
        elif key == "class_token":
            component = gr.Textbox(
                value=item["default"] if check_key(item, "default") else "",
                label="class token (use if no text prompts in reg dataset)",
                elem_id=id,
                interactive=True,
            ).style()

        elif key == "sample_prompts":
            component = gr.Textbox(
                value=item["default"] if check_key(item, "default") else "",
                label="sample prompts (seperate by '|')",
                elem_id=id,
                interactive=True,
            ).style()
        elif key == "stop_train_text_encoder":
            component = gr.Textbox(
                value=item["default"] if check_key(item, "default") else "",
                label="steps to stop training text encoder, -1 --always, 0 --never",
                elem_id=id,
                interactive=True,
            ).style()
        elif key == "max_length":
            component = gr.Textbox(
                value=item["default"] if check_key(item, "default") else "",
                label="blip: max length",
                elem_id=id,
                interactive=True,
            ).style()
        elif key == "min_length":
            component = gr.Textbox(
                value=item["default"] if check_key(item, "default") else "",
                label="blip: min length",
                elem_id=id,
                interactive=True,
            ).style()
        elif key == "general_threshold":
            component = gr.Textbox(
                value=item["default"] if check_key(item, "default") else "",
                label="tagger: general threshold (0-1)",
                elem_id=id,
                interactive=True,
            ).style()
            
        elif key == "character_threshold":
            component = gr.Textbox(
                value=item["default"] if check_key(item, "default") else "",
                label="tagger: character threshold (0-1)",
                elem_id=id,
                interactive=True,
            ).style()
        
        elif key == "undesired_tags":
            component = gr.Textbox(
                value=item["default"] if check_key(item, "default") else "",
                label="tagger: undesired tags (seperate by ',')",
                elem_id=id,
                interactive=True,
            ).style()
            continue

        elif key == "tags_to_replace":
            component = gr.Textbox(
                value=item["default"] if check_key(item, "default") else "",
                label="tags to replace (old:new, seperate by ',')",
                elem_id=id,
                interactive=True,
            ).style()

        elif key == "tags_to_add_to_front":
            component = gr.Textbox(
                value=item["default"] if check_key(item, "default") else "",
                label="tags to add to front (seperate by ',')",
                elem_id=id,
                interactive=True,
            ).style()

        elif type == list:
            choices = [
                c if c is not None else "None"
                for c in (
                    override["choices"] if "choices" in override else item["choices"]
                )
            ]
            component = gr.Radio(
                choices=choices,
                value=item["default"] if check_key(item, "default") else choices[0],
                label=key,
                elem_id=id,
                interactive=True,
            )

        elif type == bool:
            component = gr.Checkbox(
                value=item["default"] if check_key(item, "default") else False,
                label=key,
                elem_id=id,
                interactive=True,
            )
        else:
            component = gr.Textbox(
                value=item["default"] if check_key(item, "default") else "",
                label=key,
                elem_id=id,
                interactive=True,
            ).style()

        shared.help_title_map[id] = help
        out[key] = component


def args_to_gradio(args, out, overrides={}):
    options_to_gradio(args.__dict__["_option_string_actions"], out, overrides)


def gradio_to_args(arguments, options, args, strarg=False):
    def find_arg(key):
        for k, arg in arguments.items():
            arg = arg.__dict__ if hasattr(arg, "__dict__") else arg
            if arg["dest"] == key:
                return k, arg
        return None, None

    def get_value(key):
        item = args[options[key]]
        raw_key, arg = find_arg(key)
        arg_type = get_arg_type(arg)
        multiple = "nargs" in arg and arg["nargs"] == "*"

        def set_type(x):
            if x is None or x == "None":
                return None
            elif arg_type is None:
                return x
            elif arg_type == list:
                return x
            return arg_type(x)

        if multiple and item is None or item == "":
            return raw_key, None

        return raw_key, (
            [set_type(x) for x in item.split(" ")] if multiple else set_type(item)
        )

    if strarg:
        main = []
        optional = {}

        for k in options:
            key, v = get_value(k)
            if key.startswith("--"):
                key = k.replace("--", "")
                optional[key] = v
            else:
                main.append(v)

        main = [x for x in main if x is not None]

        return main, optional
    else:
        result = {}
        for k in options:
            _, v = get_value(k)
            result[k] = v
        return result


def make_args(d):
    arguments = []
    for k, v in d.items():
        if type(v) == bool:
            arguments.append(f"--{k}" if v else "")
        elif type(v) == list and len(v) > 0:
            arguments.extend([f"--{k}", *v])
        elif type(v) == str and v:
            arguments.extend([f"--{k}", f"{v}"])
        elif v:
            arguments.extend([f"--{k}", f"{v}"])
    return arguments


def run_python(script, templates, options, args):
    main, optional = gradio_to_args(templates, options, args, strarg=True)
    args = [x for x in [*main, *make_args(optional)] if x]
    proc_args = [python, "-u", script, *args]
    print("Start process: ", " ".join(proc_args))

    ps = subprocess.Popen(
        proc_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.join(ROOT_DIR, "kohya_ss_revised"),
    )
    return ps
