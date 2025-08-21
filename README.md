# Guide
## VILA
```
git clone https://github.com/NVlabs/VILA.git
cd VILA

git checkout vila1.5

apt-get update

apt-get install -y wget

bash environment_setup.sh vila15
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
{'id': 0,
 'images': ['VZ_Pearl_St_2018-03_jeh.jpg',
  'retrieved_doc_1_img.png',
  'retrieved_doc_2_img.png',
  'retrieved_doc_3_img.png',
  'retrieved_doc_4_img.png',
  'retrieved_doc_5_img.png'],
 'conversations': [
{'from': 'human',
   'value': '<image>
\nQuestion: What building is bordered by this street to the southeast and Hanover Square to the northeast?
\n
\nRetrieved passages:\n1: <image>sample document text 1
\n2: <image>sample document text 2
\n3: <image>sample document text 3
\n4: <image>sample document text 4
\n5: <image>sample document text 5
\n
\nGiven the query image and question,\nalong with retrieved passages and their
\nimages, identify the matched passages and
\nuse them to provide a short answer to the
\nquestion.'},
  {'from': 'gpt', 'value': '1 Hanover Square'}
]
}
```
