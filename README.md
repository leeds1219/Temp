# Guide
## Prerequisites

If you are using an H100 GPU, you can directly use the provided `.yml` environment files.  
For other architectures, follow the installation instructions below.

**Note:** Some fixes are required, so please install the package in editable mode.

> Due to GitHub repository size limitations, it is not possible to upload all files.  
> The following modifications have been applied manually and are included in this README.

> Some modifications may have been inadvertently missed.  
> If you encounter any issues, please feel free to open an issue on the repository, or fix them directly—your contributions are appreciated.


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
> Ensure that CUDA 11.8 (cu118) is installed. Some systems may default to CUDA 11.7, which does not support H100 (sm_90).
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
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
Notes:

> Upgrade peft and accelerate to ensure compatibility with transformers==4.49.

> FLMR originally uses transformers==4.28.

> Add num_items_in_batch=None argument to the compute_loss method in PreFLMRTrainer to comply with the updated HF Trainer API and use safe_serialization=False:
```
class PreFLMRTrainer(Trainer):
    # added , num_items_in_batch=None since transformers 4.49.0 requires this input
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        outputs = model(**inputs, return_dict=True)
        ib_loss = outputs["in_batch_negative_loss"]
        outputs["loss"] = ib_loss

        return (ib_loss, outputs) if return_outputs else ib_loss

    # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L3353C5-L3353C90
    def save_model(self, output_dir=None, _internal_call: bool = False):
        # super().save_model(output_dir, _internal_call) # need to use safe_serialization=False, option since there are shared tensor issue plus, we modified the Architecture a bit
        if output_dir is None:
            output_dir = self.args.output_dir
        
        self.query_tokenizer.save_pretrained(os.path.join(output_dir, 'query_tokenizer'), safe_serialization=False) 
        self.context_tokenizer.save_pretrained(os.path.join(output_dir, 'context_tokenizer'), safe_serialization=False)
```

## MuKA
```
git clone https://github.com/lhdeng-gh/MuKA.git
```

### To use VILA model for debugging
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

### doc_image_title2image.json Example
The code adds the root image folder to the img name.
```
{
  "WikiWeb_Heracleum mantegazzianum_11": "Hera_Campana_Louvre_Ma2283.jpg",
  "WikiWeb_Parthenocissus quinquefolia_5": "Virginia_creeper_Parthenocissus_quinquifolia_169.JPG",
  "WikiWeb_Juglans regia_1": "Noyer_centenaire_en_automne.JPG",
}
```

### questions.jsonl Example
```
{"question_id": "q1", "image": "cat.jpg", "text": "What is in the image?"}
{"question_id": "q2", "image": ["dog1.jpg", "dog2.jpg"], "text": "<image> and <image>: Are they the same breed?"}
```

### train_examples.json Example
```
# The images dir is a relative dir of both query images & document images, should place the query images in the same place as M2KR images folder
# Important! : need to truncate each document to 100 token!!! context limit issue forgot to do this in this example

{'id': 'EVQA_66',
 'images': ['inat/train/09426_Plantae_Tracheophyta_Magnoliopsida_Rosales_Rosaceae_Prunus_laurocerasus/26405447-125c-4f7c-8d90-6794214162cb.jpg',
  'images/Fr%C3%BChling_bl%C3%BChender_Kirschenbaum.jpg',
  'images/Hibiscus_Syriacus.JPG',
  'images/Enemion_biternatum_Arkansas.jpg',
  'images/Parsonsia_heterophylla_11.JPG',
  'images/WikiWeb_Annona glabra_2.jpg'],
 'conversations': [{'from': 'human',
   'value': '<image>\nQuestion: How do the of this plant seeds spread?\n\nRetrieved passages:\n1: <image> It has become naturalised widely. In some regions (such as the United Kingdom and the Pacific Northwest of North America), this species can be an invasive plant. Its rapid growth, coupled with its evergreen habit and its tolerance of drought and shade, often allow it to out-compete and kill off native plant species.  It is spread by birds, through the seeds in their droppings.\n2: <image> Most modern cultivars are virtually fruitless. The fruits of those that have them are green or brown, ornamentally unattractive 5-valved dehiscent capsules, which persist throughout much of the winter on older cultivars. They will eventually shatter over the course of the dormant season and spread their easily germinating seeds around the base of the parent plant, forming colonies with time.\n3: <image> The plant sends up evergreen basal leaves in the fall, flower stems in the spring, and goes dormant in late spring and early summer after the seed ripens.\nLeaves are twice or thrice compound with groups of three leaflets. Leaflets are smooth-edged, irregularly and deeply lobed twice or thrice, often with one to three secondary shallow lobes. Basal leaves are held on long stalks, and there are leaves arranged alternately up the flowering stems, with shorter stalks. All stems are reddish and hairless.\nThe root system is weakly rhizomatous, and occasionally produces small tubers. Plants spread over time to form thick colonies.\nThe flowering stems are 4 to 16 inches (10 to 40\xa0cm) high. Flowers are produced singly or in leafy racemes of two to four flowers, which means that there are leaves arranged alternately up the stems and flowers are in stems that come out of leaf axils. On either side of the leaf axils are two rounded stipules.\nThe flowers have five white petal-like sepals that are each 5.5–13.5\xa0mm (³⁄₁₆–⁹⁄₁₆\xa0in) long and 3.5–8.5\xa0mm (¹⁄₈–⁵⁄₁₆\xa0in) wide, 25-50 stamens with yellow pollen on the anthers, and three to six green carpels. If a carpel is fertilized, it develops into a beaked pod (follicle). When ripe, the pod splits open on one side to release several reddish-brown seeds.\n4: <image> There is little information available on the timeline and life cycle of this species. This plant flowers from September to March, followed by seed pods from February. Seeds are dispersed then by the wind.\nIf one wants to plant P. heterophylla in their garden, the optimal time to collect seeds is between February to April.\n5: <image> A. glabra thrives in wet environments. The seeds and fruit of this plant can be dispersed during wet seasons where they fall into swamps and rivers. This allows the seeds and fruits to spread to coastlines. A 2008 study found that A. glabra seeds can withstand floating in salt water and fresh water for up to 12 months. About 38% of those seeds can then germinate in soil, though A. glabra roots do not do well with constant flooding. Another study in 1998 found that even under intense flooding, the 12-month lifespan of A. glabra seedlings was unaffected; the growth rate of A. glabra trees did decrease however over a 6-month period. Compared to other Annona seeds and trees, the A. glabra is still more resilient to instances of flooding.\n\nGiven the query image and question, along with retrieved passages and their images, identify the matched passages and use them to provide a short answer to the question.'},
  {'from': 'gpt', 'value': ' by birds'}]}
```
## zero3.json Example
```
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

# Acknowledgements & Usage Disclaimer

- This project incorporates substantial code and resources from the following repositories:  
  [MuKA](https://github.com/lhdeng-gh/MuKA), [FLMR](https://github.com/LinWeizheDragon/FLMR), and [VILA 1.5](https://github.com/NVlabs/VILA). We sincerely thank the authors for their contributions.

- Additional images included in this project were collected for research purposes using publicly available sources such as Wiki APIs.  
  These images are intended solely for academic and research use. The authors of this repository do not take responsibility for any other usage.

- All collected images, datasets, and related files will be made available at:  
  [Here](https://huggingface.co/datasets/Lee1219/WikiCrawl_2025_08_19)
- The paths for images, datasets, and models will be organized and documented soon.  
  Due to time constraints, a complete organization has not yet been done, but the resources will be made available as soon as possible.  
  (Currently, some unorganized files are included for convenience.)

