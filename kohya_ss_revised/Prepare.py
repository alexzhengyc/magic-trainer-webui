import os
import argparse
import re
from PIL import Image
import random
import concurrent.futures
import subprocess
from tqdm import tqdm
    
class Prepare:
    def __init__(self, **kwargs):
        self.anotation_method = kwargs.get("anotation_method", "blip") 
        self.train_data_dir = kwargs.get("train_data_dir", "")
        self.recursive = kwargs.get("recursive", False)
        self.batch_size = kwargs.get("batch_size", 2)
        self.top_p = kwargs.get("top_p", 0.9)
        self.blip = kwargs.get("blip", "")
        self.max_length = kwargs.get("max_length", 75)
        self.min_length = kwargs.get("min_length", 5)
        self.threshold = kwargs.get("threshold", 0.5)
        self.general_threshold = kwargs.get("general_threshold", 0.5)
        self.character_threshold = kwargs.get("character_threshold", 0.5)
        self.undesired_tags = kwargs.get("undesired_tags", "")
        self.tags_to_add_to_front = kwargs.get("tags_to_add_to_front", "")
        self.tags_to_replace = kwargs.get("tags_to_replace", "")


    def run(self):
        kohya_dir = os.path.dirname(os.path.realpath(__file__))
        magic_trainer_dir = os.path.dirname(kohya_dir)
        extensions_dir = os.path.dirname(magic_trainer_dir)
        stable_diffusion_dir = os.path.dirname(extensions_dir)
        blip_dir = os.path.join(stable_diffusion_dir, "models/BLIP")
        finetune_dir = os.path.join(kohya_dir, "finetune")

        if not self.blip.endswith((".pt", ".pth", ".bin", ".ckpt", ".safetensors")):
            self.blip_path = None
        else:
            self.blip_path = self.blip

        convert = True  # @param {type:"boolean"}
        random_color = False  # @param {type:"boolean"}
        batch_size = 32
        images = [
            image
            for image in os.listdir(self.train_data_dir)
            if image.endswith(".png") or image.endswith(".webp")
        ]
        background_colors = [
            (255, 255, 255),
            (0, 0, 0),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        def process_image(image_name):
            img = Image.open(f"{self.train_data_dir}/{image_name}")

            if img.mode in ("RGBA", "LA"):
                if random_color:
                    background_color = random.choice(background_colors)
                else:
                    background_color = (255, 255, 255)
                bg = Image.new("RGB", img.size, background_color)
                bg.paste(img, mask=img.split()[-1])

                if image_name.endswith(".webp"):
                    bg = bg.convert("RGB")
                    bg.save(
                        f'{self.train_data_dir}/{image_name.replace(".webp", ".jpg")}',
                        "JPEG",
                    )
                    os.remove(f"{self.train_data_dir}/{image_name}")
                    print(
                        f" Converted image: {image_name} to {image_name.replace('.webp', '.jpg')}"
                    )
                else:
                    bg.save(f"{self.train_data_dir}/{image_name}", "PNG")
                    print(f" Converted image: {image_name}")
            else:
                if image_name.endswith(".webp"):
                    img.save(
                        f'{self.train_data_dir}/{image_name.replace(".webp", ".jpg")}',
                        "JPEG",
                    )
                    os.remove(f"{self.train_data_dir}/{image_name}")
                    print(
                        f" Converted image: {image_name} to {image_name.replace('.webp', '.jpg')}"
                    )
                else:
                    img.save(f"{self.train_data_dir}/{image_name}", "PNG")

        num_batches = len(images) // batch_size + 1
        if convert:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in tqdm(range(num_batches)):
                    start = i * batch_size
                    end = start + batch_size
                    batch = images[start:end]
                    executor.map(process_image, batch)

            print("All images have been converted")
        
        os.chdir(finetune_dir)
        # ## Data Annotation
        # You can choose to train a model using captions. We're using [BLIP](https://huggingface.co/spaces/Salesforce/BLIP) for image captioning and [Waifu Diffusion 1.4 Tagger](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags) for image tagging similar to Danbooru.
        # - Use BLIP Captioning for: `General Images`
        # - Use Waifu Diffusion 1.4 Tagger V2 for: `Anime and Manga-style Images`
        if self.anotation_method == "blip" or self.anotation_method == "both":
            max_data_loader_n_workers = 2  # @param {type:'number'}
            beam_search = True  # @param {type:'boolean'}
            if self.blip_path is not None:
                command = f"""python make_captions.py "{self.train_data_dir}" --caption_weights {self.blip_path} --batch_size {self.batch_size} {"--beam_search" if beam_search else ""} --min_length {self.min_length} --max_length {self.max_length} --caption_extension .txt --max_data_loader_n_workers {max_data_loader_n_workers}"""
            else:
                command = f"""python make_captions.py "{self.train_data_dir}" --batch_size {self.batch_size} {"--beam_search" if beam_search else ""} --min_length {self.min_length} --max_length {self.max_length} --caption_extension .txt --max_data_loader_n_workers {max_data_loader_n_workers}"""
            subprocess.run(command, shell=True, check=True)

        # 4.2.2. Waifu Diffusion 1.4 Tagger V2

        if self.anotation_method == "wd14-tagger":
            max_data_loader_n_workers = 2  # @param {type:'number'}
            model = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"  # @param ["SmilingWolf/wd-v1-4-convnextv2-tagger-v2", "SmilingWolf/wd-v1-4-swinv2-tagger-v2", "SmilingWolf/wd-v1-4-convnext-tagger-v2", "SmilingWolf/wd-v1-4-vit-tagger-v2"]
            # @markdown Use the `recursive` option to process subfolders as well, useful for multi-concept training.
            # @markdown Debug while tagging, it will print your image file with general tags and character tags.
            verbose_logging = False  # @param {type:"boolean"}
            # @markdown Separate `undesired_tags` with comma `(,)` if you want to remove multiple tags, e.g. `1girl,solo,smile`.

            config = {
                "_train_data_dir": self.train_data_dir,
                "batch_size": self.batch_size,
                "repo_id": model,
                "recursive": self.recursive,
                "remove_underscore": True,
                "general_threshold": self.general_threshold,
                "character_threshold": self.character_threshold,
                "caption_extension": ".txt",
                "max_data_loader_n_workers": max_data_loader_n_workers,
                "debug": verbose_logging,
                "undesired_tags": self.undesired_tags,
            }

            args = ""
            for k, v in config.items():
                if k.startswith("_"):
                    args += f'"{v}" '
                elif isinstance(v, str):
                    args += f'--{k}="{v}" '
                elif isinstance(v, bool) and v:
                    args += f"--{k} "
                elif isinstance(v, float) and not isinstance(v, bool):
                    args += f"--{k}={v} "
                elif isinstance(v, int) and not isinstance(v, bool):
                    args += f"--{k}={v} "

            final_args = f"python tag_images_by_wd14_tagger.py {args}"
            subprocess.run(final_args, shell=True, check=True)

        def read_file(file_path):
            with open(file_path, "r") as file:
                content = file.read()
            return content

        def remove_redundant_words(content1, content2):
            return content1.rstrip('\n') + ', ' + content2

        def write_file(file_path, content):
            with open(file_path, "w") as file:
                file.write(content)

        ### Combine BLIP and Waifu
        if self.anotation_method == "both":
            max_data_loader_n_workers = 2  # @param {type:'number'}
            model = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"  # @param ["SmilingWolf/wd-v1-4-convnextv2-tagger-v2", "SmilingWolf/wd-v1-4-swinv2-tagger-v2", "SmilingWolf/wd-v1-4-convnext-tagger-v2", "SmilingWolf/wd-v1-4-vit-tagger-v2"]
            # @markdown Use the `recursive` option to process subfolders as well, useful for multi-concept training.
            # @markdown Debug while tagging, it will print your image file with general tags and character tags.
            verbose_logging = False  # @param {type:"boolean"}
            # @markdown Separate `undesired_tags` with comma `(,)` if you want to remove multiple tags, e.g. `1girl,solo,smile`.

            config = {
                "_train_data_dir": self.train_data_dir,
                "batch_size": self.batch_size,
                "repo_id": model,
                "recursive": self.recursive,
                "remove_underscore": True,
                "general_threshold": self.general_threshold,
                "character_threshold": self.character_threshold,
                "caption_extension": ".newtxt",
                "max_data_loader_n_workers": max_data_loader_n_workers,
                "debug": verbose_logging,
                "undesired_tags": self.undesired_tags,
            }

            args = ""
            for k, v in config.items():
                if k.startswith("_"):
                    args += f'"{v}" '
                elif isinstance(v, str):
                    args += f'--{k}="{v}" '
                elif isinstance(v, bool) and v:
                    args += f"--{k} "
                elif isinstance(v, float) and not isinstance(v, bool):
                    args += f"--{k}={v} "
                elif isinstance(v, int) and not isinstance(v, bool):
                    args += f"--{k}={v} "

            final_args = f"python tag_images_by_wd14_tagger.py {args}"
            subprocess.run(final_args, shell=True, check=True)

            def combine():
                directory = self.train_data_dir
                extension1 = ".txt"
                extension2 = ".newtxt"
                output_extension = ".txt"

                for file in os.listdir(directory):
                    if file.endswith(extension1):
                        filename = os.path.splitext(file)[0]
                        file1 = os.path.join(directory, filename + extension1)
                        file2 = os.path.join(directory, filename + extension2)
                        output_file = os.path.join(directory, filename + output_extension)

                        if os.path.exists(file2):
                            content1 = read_file(file1)
                            content2 = read_file(file2)
                            combined_content = remove_redundant_words(content1, content2)
                            write_file(output_file, combined_content)
                            os.remove(file2)

            combine()

        def add_tag_to_front(filename, tags):
            contents = read_file(filename)

            # add the tag
            tags = "".join(tags.split(", "))
            contents = tags + ", " + contents
            write_file(filename, contents)

        def replace_tag(filename, tags):
            contents = read_file(filename)

            # process the tags input into a dictionary
            tag_dict = {}
            tag_pairs = tags.split(", ")
            for pair in tag_pairs:
                old, new = pair.split(":")
                tag_dict[old.strip()] = new.strip()

            # replace tags in the contents
            for old, new in tag_dict.items():
                contents = re.sub(r'\b' + old + r'\b', new, contents)
            write_file(filename, contents)

        if self.tags_to_replace != "":
            for filename in os.listdir(self.train_data_dir):
                if filename.endswith(".txt"):
                    file_path = os.path.join(self.train_data_dir, filename)
                    replace_tag(file_path, self.tags_to_replace)   

        if self.tags_to_add_to_front != "":
            for filename in os.listdir(self.train_data_dir):
                if filename.endswith(".txt"):
                    file_path = os.path.join(self.train_data_dir, filename)
                    add_tag_to_front(file_path, self.tags_to_add_to_front)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anotation_method", type=str, default="blip", choices=["blip", "wd14-tagger", "both"], help="anotation method")
    
    parser.add_argument("--train_data_dir", type=str, default="", help="directory for train images")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size in inference")

    # parser.add_argument("--num_beams", type=int, default=1, help="num of beams in beam search ")
    # parser.add_argument("--top_p", type=float, default=0.9, help="top_p in Nucleus sampling ")
    parser.add_argument("--blip", type=str, help="path to blip")
    parser.add_argument("--max_length", type=int, default=75, help="max length of caption ")
    parser.add_argument("--min_length", type=int, default=5, help="min length of caption ")
    # parser.add_argument("--threshold", type=float, default=0.35, help="threshold of confidence to add a tag ")
    parser.add_argument(
        "--general_threshold",
        type=float,
        default=0.3,
        help="threshold of confidence to add a tag for general category, same as --thresh if omitted ",
    )
    parser.add_argument(
        "--character_threshold",
        type=float,
        default=0.3,
        help="threshold of confidence to add a tag for character category, same as --thres if omitted ",
    )
    # parser.add_argument("--recursive", action="store_true", help="search for images in subfolders recursively")

    parser.add_argument(
        "--undesired_tags",
        type=str,
        default="",
        help="comma-separated list of undesired tags to remove from the output",
    )

    parser.add_argument(
        "--tags_to_replace",
        type=str,
        default="",
        help="comma-separated list of tags to delete from the output",
    )
    parser.add_argument(
        "--tags_to_add_to_front",
        type=str,
        default="",
        help="comma-separated list of tags to delete from the output",
    )


    return parser

if __name__ == "__main__":
    parser = setup_parser()
    
    args = parser.parse_args()

    if args.general_threshold is None:
        args.general_threshold = args.thresh
    if args.character_threshold is None:
        args.character_threshold = args.thresh
    config = vars(args)
    prepare = Prepare(**config)
    prepare.run()
