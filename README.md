# Guide
## VILA
```
git clone https://github.com/NVlabs/VILA.git
cd VILA

git checkout vila1.5

apt-get update

apt-get install -y wget

bash environment_setup.sh vila15

pip install numpy==1.26.4
```

## FLMR
```
git clone https://github.com/LinWeizheDragon/FLMR

conda create -n FLMR python=3.10 -y
conda activate FLMR

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl

cd FLMR
pip install -e .

cd third_party/ColBERT
pip install -e .

pip install ujson gitpython easydict ninja datasets transformers==4.49
```
## MuKA
```
git clone https://github.com/lhdeng-gh/MuKA.git
```

### To use VILA model
```
from vila_diffs.llava.model.builder import load_pretrained_model

def load_pretrained_model(
    model_path,
    model_name,
    model_base=None,
    load_8bit=False,
    load_4bit=False,
    device_map="auto", # the function overides the device_map argument internally, so this argument basically does not work!!!
    device="cuda",
    **kwargs,
)

def prepare_config_for_eval(config: PretrainedConfig, kwargs: dict):
    try:
        # compatible with deprecated config convention
        if getattr(config, "vision_tower_cfg", None) is None:
            config.vision_tower_cfg = config.mm_vision_tower
    except AttributeError:
        raise ValueError(f"Invalid configuration! Cannot find vision_tower in config:\n{config}")
    
    config.model_dtype = kwargs.pop("torch_dtype").__str__()
    # siglip does not support device_map = "auto"
    vision_tower_name = parse_model_name_or_path(config, "vision_tower")
    if "siglip" in vision_tower_name.lower():
        kwargs["device_map"] = "cuda" # TODO: should fix this

# instead just use this for debug
model_path = "/workspace/MuKA/VILA1.5-13b"
config = AutoConfig.from_pretrained(model_path)
config.resume_path = model_path
model = LlavaLlamaModel(
    config=config,
    low_cpu_mem_usage=True,
    device_map="cuda:5"
)
tokenizer = model.tokenizer
```

### doc_image_title2image.json should look like
```
{
  "WikiWeb_Heracleum mantegazzianum_11": "Hera_Campana_Louvre_Ma2283.jpg",
  "WikiWeb_Parthenocissus quinquefolia_5": "Virginia_creeper_Parthenocissus_quinquifolia_169.JPG",
  "WikiWeb_Juglans regia_1": "Noyer_centenaire_en_automne.JPG",
}
```

### train_examples.json should look like
```
train_examples[0] = {
    'id': 0,
    'images': [
        'VZ_Pearl_St_2018-03_jeh.jpg',
        'retrieved_doc_1_img.png',
        'retrieved_doc_2_img.png',
        'retrieved_doc_3_img.png',
        'retrieved_doc_4_img.png',
        'retrieved_doc_5_img.png'
    ],
    'conversations': [
        {
            'from': 'human',
            'value': (
                '<image>\n'
                'Question: What building is bordered by this street to the southeast and '
                'Hanover Square to the northeast?\n\n'
                'Retrieved passages:\n'
                '1: <image> sample document text 1\n'
                '2: <image> sample document text 2\n'
                '3: <image> sample document text 3\n'
                '4: <image> sample document text 4\n'
                '5: <image> sample document text 5\n\n'
                'Given the query image and question, along with retrieved passages and their images, '
                'identify the matched passages and use them to provide a short answer to the question.'
            )
        },
        {
            'from': 'gpt',
            'value': '1 Hanover Square'
        }
    ]
}
```
