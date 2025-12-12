# HGATv2: Interpretable Heterogeneous Graph Attention Network for High-Value Patent Identification

## Description

This repository contains the implementation of **HGATv2**, an interpretable heterogeneous graph attention network for high-value patent identification. The model integrates multiple textual fields of patents with structural information (e.g., word co-occurrence, patent classes, inter-document similarity) and performs binary classification of high-value vs. non-high-value patents. All experiments reported in the associated article can be reproduced using the code and instructions provided here.

---

## Dataset Information

### 1. Proprietary lithography patent dataset (IncoPat)

We construct a lithography patent dataset from the **IncoPat Global Patent Database** (HeXiang XinChuang Technology Co., Ltd., Beijing, China; https://www.incopat.com/), which serves as the source of Chinese-language patent records. We restrict the search to Chinese patents (country code "CN") and use a keyword-based query targeting lithography technologies, key equipment and specific lithography techniques. The search returns 115,386 lithography-related patents.

IncoPat provides a proprietary patent value score, referred to as the **HeXiang value score**, ranging from 1 to 10. We use this score, together with patent validity and legal status, to construct a binary dataset for high-value patent identification:

- High-value patents (positive class): HeXiang value score = 10, validity = valid, legal status = granted (1,524 samples).  
- Non-high-value patents (negative class): HeXiang value score = 1, validity = invalid, legal status in rejected / withdrawn / abandoned due to double-filing / lapsed due to non-payment of fees (968 samples).

The final lithography dataset used in our experiments therefore contains **2,492** patents (1,524 high-value and 968 non-high-value records). Due to licensing restrictions on IncoPat, the raw patent data cannot be redistributed here; however, the database name, query conditions and labeling rules are described in detail in the article.

### 2. Public patent dataset (PatentsView)

We also use a public patent dataset derived from the **PatentsView** platform (https://www.patentsview.org/), which is built on data from the United States Patent and Trademark Office (USPTO). PatentsView provides structured metadata including patent identifiers, invention titles, Cooperative Patent Classification (CPC) codes and abstracts. The PatentsView bulk data are archived in the Datalumos/ICPSR Dataverse under the “Patents View” project (DOI: 10.3886/E223582V1).

Based on the bulk-download table **g_cpc_title**, we construct a subset named **g_cpc_title_10000** by removing records with missing CPC information and randomly sampling **10,000** granted patents with complete CPC group codes and titles. We restrict the sample to **12** distinct CPC subclasses to ensure topic diversity. In this dataset, the invention title is used as the input text and the CPC subclass code is used as the target label.

---

## Code Information

The main script in this repository is:

- `HGATv2.py` – loads patent data, builds a heterogeneous graph using DGL, defines the HGATv2 model and runs stratified k-fold cross-validation, outputting classification metrics and plots.

The script expects:

- An Excel file containing the patent records (see Data schema below).  
- A pretrained Chinese word embedding file (e.g., SGNS/word2vec in text format).  
- A Chinese stopword list.

### Data schema (Excel file for lithography dataset)

The Excel file passed via `--train_xlsx` should minimally contain:

- `value_score` – HeXiang value score (integer 1–10).  
- `title` – Patent title.  
- `abstract` – Patent abstract.  
- `independent_claim` – Independent claim text.  
- `technical_effect_sentences` – Sentences describing the technical effect or advantages.  

Users may adapt the field names in the script if different column names are used.

---

Usage Instructions

Set up a Python environment (e.g., using Conda or a virtual environment) with Python 3.8 or higher.

Install the dependencies listed in the Requirements section (e.g., torch, dgl, numpy, pandas, jieba, gensim, scikit-learn, rank-bm25, matplotlib).

Prepare an Excel file containing the lithography patent dataset with the required columns (value_score, title, abstract, independent_claim, technical_effect_sentences, etc.).

Prepare a pretrained Chinese word embedding file in text format (such as an SGNS/word2vec model) and a Chinese stopword list.

Run the main script HGATv2.py in your Python environment, specifying at least:

the path to the Excel file (argument --train_xlsx),

the path to the word embedding file (argument --vector_path), and

the computing device (argument --device, set to cpu or cuda).

After execution, the script reports performance for each fold of cross-validation and saves model checkpoints and plots in the models/ and plots/ directories.



Environment setup
Create and activate a Python environment (example using Conda):

```bash
conda create -n hgatv2_env python=3.9
conda activate hgatv2_env
Install dependencies:pip install torch dgl numpy pandas jieba gensim scikit-learn rank-bm25 matplotlib
(Alternatively, install from requirements.txt if provided.)



Key arguments:
--train_xlsx: path to the Excel file with patent records.
--vector_path: path to the pretrained word embedding file (text format).
--device: cuda or cpu.
After training, the script prints per-fold metrics and saves model checkpoints and plots under models/ and plots/.



Requirements
Python 3.8+ (tested with Python 3.9)
Core libraries:
torch
dgl
numpy, pandas
jieba, gensim
scikit-learn, rank-bm25
matplotlib
A minimal requirements.txt can list these packages and their versions.


Methodology 

At a high level, the implementation follows these steps:

Load patent data and construct binary labels for high-value vs. non-high-value patents using the HeXiang value score and validity/legal status rules.

Perform Chinese word segmentation and stopword removal on multiple text fields (title, abstract, independent claim, technical-effect sentences).

Map tokens to pretrained word embeddings and encode text fields with a CNN + attention encoder.

Build a heterogeneous graph with document, sentence and word nodes, and multiple edge types (document–text, sentence–word, word–word co-occurrence, document–word, document–document).

Apply a heterogenous GATv2-based model to learn document embeddings and predict the binary label.

Train and evaluate the model using stratified k-fold cross-validation and report accuracy, precision, recall, F1 and ROC-AUC.

For full methodological details, please refer to the associated article.



 Citations

If you use this code or the associated datasets in your research, please cite:

Fh.X, et al. Interpretable heterogeneous graph attention networks for high-value patent identification. Manuscript submitted to PeerJ Computer Science, 2025.



Data sources:
- IncoPat Global Patent Database (HeXiang XinChuang Technology Co., Ltd., Beijing, China; https://www.incopat.com/).
- PatentsView platform and its archived bulk data (Patents View project, Datalumos/ICPSR Dataverse, DOI: 10.3886/E223582V1).




License

This code is released under the MIT License (or another OSI-approved license you choose). Please include a LICENSE file that matches the statement here.




Contributions

This repository is primarily intended to support reproducibility of the corresponding article. Bug reports and small improvements are welcome via issue or pull request, provided that changes are clearly documented and do not conflict with the original experimental setup.