---
library_name: transformers
tags:
- bulk RNA-seq
- biology
- transcriptomics
---

# BulkRNABert

BulkRNABert is a transformer-based, encoder-only language model pre-trained on bulk RNA-seq profiles from the TCGA dataset using self-supervised masked language modeling, following the original BERT framework. The model is trained to reconstruct randomly masked gene expression values from their genomic context, enabling it to learn biologically meaningful representations of transcriptomic profiles. Once pre-trained, BulkRNABert can be fine-tuned for various cancer-related downstream tasks—such as cancer type classification or survival analysis—by extracting embeddings from the model.

**Developed by:** [InstaDeep](https://huggingface.co/InstaDeepAI)

### Model Sources

<!-- Provide the basic links for the model. -->

- [**Repository**](https://github.com/instadeepai/multiomics-open-research)
- **Paper:** [BulkRNABert: Cancer prognosis from bulk RNA-seq based language models](https://proceedings.mlr.press/v259/gelard25a.html)

### How to use

Until its next release, the transformers library needs to be installed from source using the following command to use the models.
PyTorch should also be installed.

```
pip install --upgrade git+https://github.com/huggingface/transformers.git
pip install torch
```

## Other notes
We also provide the params for the BulkRNABert jax model in `jax_params`.

A small snippet of code is provided below to run inference with the model using bulk RNA-seq samples from the [TCGA](https://portal.gdc.cancer.gov/) dataset.

```
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Load model and tokenizer.
config = AutoConfig.from_pretrained(
    "InstaDeepAI/BulkRNABert",
    trust_remote_code=True,
)
config.embeddings_layers_to_save = (4,) # last transformer layer

tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/BulkRNABert", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "InstaDeepAI/BulkRNABert",
    config=config,
    trust_remote_code=True,
)

# Load bulk RNA-seq data and preprocess them.
csv_path = hf_hub_download(
    repo_id="InstaDeepAI/BulkRNABert",
    filename="data/tcga_sample.csv",
    repo_type="model",
)
gene_expression_array = pd.read_csv(csv_path).drop(["identifier"], axis=1).to_numpy()[:1, :]
gene_expression_array = np.log10(1 + gene_expression_array)
assert gene_expression_array.shape[1] == config.n_genes

# Tokenize
gene_expression_ids = tokenizer.batch_encode_plus(gene_expression_array, return_tensors="pt")["input_ids"]

# Compute BulkRNABert's embeddings
gene_expression_mean_embeddings = model(gene_expression_ids)["embeddings_4"].mean(axis=1)  # embeddings can be used for downstream tasks.
```


### Citing our work

```
@InProceedings{pmlr-v259-gelard25a,
  title = 	 {BulkRNABert: Cancer prognosis from bulk RNA-seq based language models},
  author =       {G{\'{e}}lard, Maxence and Richard, Guillaume and Pierrot, Thomas and Courn{\`{e}}de, Paul-Henry},
  booktitle = 	 {Proceedings of the 4th Machine Learning for Health Symposium},
  pages = 	 {384--400},
  year = 	 {2025},
  editor = 	 {Hegselmann, Stefan and Zhou, Helen and Healey, Elizabeth and Chang, Trenton and Ellington, Caleb and Mhasawade, Vishwali and Tonekaboni, Sana and Argaw, Peniel and Zhang, Haoran},
  volume = 	 {259},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {15--16 Dec},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v259/gelard25a.html},
}
```
