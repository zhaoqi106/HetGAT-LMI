# Reproducibility and Environment

## 1) Software Versions (pinned)
- Python: 3.8.10
- CUDA Toolkit (nvcc): 11.5
- torch==1.13.0
- torch-geometric==2.6.1
- numpy==1.23.5
- pandas==1.3.5
- scikit-learn==1.3.2
- scipy==1.10.1
- matplotlib==3.7.5
- seaborn==0.12.2
- captum==0.7.0
- shap==0.44.1
- networkx==3.1
- biopython==1.83
- gensim==4.3.3
- karateclub==1.3.3
- node2vec==0.5.0
- stellargraph==1.2.1
- catboost==1.2.8
- lightgbm==4.6.0
- tensorflow==2.13.0

External dependency:
ViennaRNA / RNAfold (install via system package manager or official installer; ensure the RNAfold command is available)

Hardware note: Experiments were run on an NVIDIA RTX 3090 (24 GB VRAM). The pipeline can also run on CPU, but training and feature extraction will be slower.

## 2) Installation 
```bash
python -m venv .venv
.\.venv\Scripts\activate

pip install --upgrade pip==25.0.1
pip install torch==1.13.0
pip install torch-geometric==2.6.1
pip install numpy==1.23.5 pandas==1.3.5 scikit-learn==1.3.2 scipy==1.10.1
pip install matplotlib==3.7.5 seaborn==0.12.2
pip install captum==0.7.0 shap==0.44.1 networkx==3.1 biopython==1.83
pip install gensim==4.3.3 karateclub==1.3.3 node2vec==0.5.0 stellargraph==1.2.1
pip install catboost==1.2.8 lightgbm==4.6.0

## 3) Usage
Full pipeline (sequence → structure/RNAfold → fusion → graph → train)
cd /d D:\pytorchProject\HetGAT-LMI\code
    python feature_ex.py
    python feature_2.py
    python fusion.py
    python edge.py
    python hg_graph.py
    python main.py
