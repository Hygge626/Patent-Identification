import argparse
import os
from collections import defaultdict
from typing import List, Dict
import nltk
nltk.data.path.append('nltk_data_local')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn import GATv2Conv, HeteroGraphConv
try:
    from dgl.dataloading import NodeDataLoader
except ImportError:
    from dgl.dataloading import DataLoader as NodeDataLoader
from dgl.dataloading import MultiLayerNeighborSampler, MultiLayerFullNeighborSampler

from nltk.tokenize import word_tokenize

import sys
from datetime import datetime


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/Repair_batch_size_mismatch_{timestamp}.log"
os.makedirs("logs", exist_ok=True)

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, open(log_file, 'w', encoding='utf-8'))
sys.stderr = sys.stdout


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to {path}")

def smooth(values, window=3):
    if len(values) < window:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(values[start:i+1]))
    return smoothed

def save_metric_plot(train_values, val_values, ylabel, out_path):
    plt.figure()
    plt.plot(smooth(train_values), label='Train')
    plt.plot(smooth(val_values), label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} Curve')
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def load_sgns(path: str) -> KeyedVectors:
    print(f"[INFO] Loading embeddings from {path} …")
    kv = KeyedVectors.load_word2vec_format(path, binary=False)
    print(f"[INFO] Loaded {len(kv)} tokens, dim={kv.vector_size}")
    return kv

class StaticDeltaEmbedding(nn.Module):
    def __init__(self, kv: KeyedVectors):
        super().__init__()
        self.kv = kv
        self.dim = kv.vector_size
        self.delta = nn.ParameterDict()

    def _safe_key(self, token: str) -> str:
        return f"tok_{token.replace('.', '_dot_').replace('/', '_slash_').replace(' ', '_')}"

    def forward(self, token: str) -> torch.Tensor:
        base = torch.tensor(
            self.kv[token] if token in self.kv else np.random.normal(0, 0.05, self.dim),
            dtype=torch.float
        )
        key = self._safe_key(token)
        if key not in self.delta:
            self.delta[key] = nn.Parameter(torch.zeros(self.dim))
        return base + self.delta[key]


class CNNAttentionEncoder(nn.Module):
    def __init__(self, emb_dim: int, channels: int = 250, kernel_sizes: List[int] = [3,5,7], max_len: int = 512):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, channels, k, padding=k//2)
            for k in kernel_sizes
        ])
        heads = min(4, len(kernel_sizes))
        while heads > 1 and channels % heads != 0:
            heads -= 1
        self.attention = nn.MultiheadAttention(channels, num_heads=heads, batch_first=True)
        self.out_dim = channels
        self.position_encoding = nn.Parameter(torch.randn(1, max_len, emb_dim))

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        L, D = embeds.shape
        pe = self.position_encoding[:, :L, :] if L <= self.position_encoding.size(1) \
            else torch.cat([self.position_encoding, torch.zeros(1, L - self.position_encoding.size(1), D, device=embeds.device)], dim=1)
        x = embeds.unsqueeze(0) + pe
        x = x.transpose(1, 2)
        convs = [F.relu(conv(x)) for conv in self.convs]
        pools = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in convs]
        seq = torch.stack(pools, dim=1)
        attn_out, _ = self.attention(seq, seq, seq)
        return attn_out.mean(1).squeeze(0)

class PatentGraphBuilder:
    def __init__(self, embedder: StaticDeltaEmbedding, encoder: CNNAttentionEncoder):
        self.embedder = embedder
        self.encoder = encoder
        self.node_feats = defaultdict(list)
        self.node_offsets = defaultdict(int)
        self.word2nid: Dict[str, int] = {}
        self.edges = defaultdict(list)
        self.doc_nids, self.doc_labels = [], []

    def reset(self):
        self.node_feats.clear()
        self.node_offsets.clear()
        self.word2nid.clear()
        self.edges.clear()
        self.doc_nids.clear()
        self.doc_labels.clear()

    def _add_node(self, ntype: str, feat: torch.Tensor) -> int:
        nid = self.node_offsets[ntype]
        self.node_offsets[ntype] += 1
        self.node_feats[ntype].append(feat)
        return nid

    def _get_word(self, tok: str) -> int:
        if tok not in self.word2nid:
            nid = self._add_node('word', self.embedder(tok).detach())
            self.word2nid[tok] = nid
        return self.word2nid[tok]

    def _encode(self, toks: List[str]) -> torch.Tensor:
        if not toks:
            toks = ['<pad>']
        embeds = torch.stack([self.embedder(t) for t in toks])
        return self.encoder(embeds)

    def build(self, df: pd.DataFrame):
        texts = df['cpc_subclass_title'].fillna('').astype(str).tolist()
        tokens_list = [word_tokenize(text) for text in texts]

        tfidf_vec = TfidfVectorizer()
        tfidf_matrix = tfidf_vec.fit_transform(texts)
        bm25 = BM25Okapi(tokens_list)
        bm25_scores = [bm25.get_scores(tokens)[i] for i, tokens in enumerate(tokens_list)]

        for i, toks in enumerate(tokens_list):
            label = int(df['label'].iloc[i])
            doc_id = self._add_node('document', torch.zeros(self.encoder.out_dim))
            self.doc_nids.append(doc_id)
            self.doc_labels.append(label)

            title_id = self._add_node('title', self._encode(toks))
            self.edges[('document', 'has_title', 'title')].append((doc_id, title_id))

            for tok in toks:
                w = self._get_word(tok)
                self.edges[('title', 'has_word', 'word')].append((title_id, w))

            row = tfidf_matrix.getrow(i)
            for idx in row.indices:
                token = tfidf_vec.get_feature_names_out()[idx]
                if token in self.word2nid:
                    self.edges[('document', 'has_tfidf_word', 'word')].append((doc_id, self.word2nid[token]))

        cpc_map = defaultdict(list)
        for idx, cls in enumerate(df['cpc_class'].fillna('').astype(str)):
            cpc_map[cls.strip()].append(idx)
        for docs in cpc_map.values():
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    self.edges[('document', 'cpc_class_sim', 'document')].append((docs[i], docs[j]))

        rev = {}
        for (src, et, dst), pairs in list(self.edges.items()):
            rev[(dst, f"rev_{et}", src)] = [(d, s) for s, d in pairs]
        for k, v in rev.items():
            self.edges[k].extend(v)

        g_data = {
            etype: (
                torch.tensor([s for s, _ in pairs], dtype=torch.int64),
                torch.tensor([d for _, d in pairs], dtype=torch.int64)
            )
            for etype, pairs in self.edges.items() if pairs
        }
        g = dgl.heterograph(g_data)

        for ntype, feats in self.node_feats.items():
            g.nodes[ntype].data['h'] = torch.stack(feats).detach()
        g.nodes['document'].data['label'] = torch.tensor(self.doc_labels)
        g.nodes['document'].data['bm25'] = torch.tensor(bm25_scores, dtype=torch.float).unsqueeze(1)

        print("[INFO] Edge type summary:")
        for k, v in self.edges.items():
            print(f"  {k}: {len(v)} edges")

        return g, self.doc_nids

def build_fold_graph(train_df, val_df, embedder, encoder):
    df_combined = pd.concat([train_df, val_df], ignore_index=True)
    builder = PatentGraphBuilder(embedder, encoder)
    g, doc_nids = builder.build(df_combined)
    return g, doc_nids


class HeteroGATv2(nn.Module):
    def __init__(self, graph:dgl.DGLHeteroGraph, in_dim, hidden_dim, heads, layers):
        super().__init__()
        self.feat_project = nn.ModuleDict({
            n: nn.Linear(300 if n=='word' else in_dim, in_dim)
            for n in graph.ntypes
        })
        self.layers = nn.ModuleList()
        for l in range(layers):
            convs = {}
            for src, et, dst in graph.canonical_etypes:
                convs[(src,et,dst)] = GATv2Conv(
                    in_dim if l==0 else hidden_dim*heads,
                    hidden_dim, heads,
                    feat_drop=0.5, attn_drop=0.5,
                    allow_zero_in_degree=True
                )
            self.layers.append(HeteroGraphConv(convs, aggregate='mean'))
        self.clf = nn.Sequential(
            nn.Linear(hidden_dim*heads, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, len_label)
        )

    def forward(self, blocks, h_dict):
        h = {nt: self.feat_project[nt](feat) for nt, feat in h_dict.items()}
        for layer, blk in zip(self.layers, blocks):
            out = layer(blk, h)
            h = {k: v.flatten(1) for k, v in out.items()}
        if 'document' not in h or h['document'].size(0) == 0:
            print("[WARN] No document nodes in h. Returning empty logits.")
            return torch.zeros((0, len_label), device=next(self.parameters()).device)
        dst_ids = blocks[-1].dstnodes['document'].data[dgl.NID]
        logits_raw = self.clf(h['document'])
        logits_dict = {nid.item(): logit for nid, logit in zip(dst_ids, logits_raw)}
        aligned_logits = torch.stack([
            logits_dict.get(nid.item(), torch.zeros(len_label, device=next(self.parameters()).device))
            for nid in dst_ids
        ])
        print(f"[DEBUG] Batch logits shape: {aligned_logits.shape}, labels shape: {dst_ids.shape}")
        return aligned_logits


def get_metrics(y_true:torch.Tensor, y_prob:torch.Tensor):
    y_pred = y_prob.argmax(1)
    return {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'auc': roc_auc_score(F.one_hot(y_true, num_classes=y_prob.size(1)), y_prob, multi_class='ovo'),
        'cm': confusion_matrix(y_true, y_pred),
        'report': classification_report(y_true, y_pred, digits=3)
    }

def evaluate(model, g, nids, sampler, device, loss_fn):
    model.eval()
    loader = NodeDataLoader(
        g, {'document': torch.tensor(nids)}, sampler,
        batch_size=512, shuffle=False, num_workers=0
    )
    y_true, y_prob, losses = [], [], []
    with torch.no_grad():
        for _, _, blocks in loader:
            blocks = [b.to(device) for b in blocks]
            feats  = {nt: blocks[0].srcnodes[nt].data['h'] for nt in blocks[0].srctypes}
            labels = blocks[-1].dstnodes['document'].data['label']
            logits = model(blocks, feats)
            if logits.size(0) != labels.size(0):
                print(f"[ERROR] Mismatch: logits={logits.size(0)} labels={labels.size(0)}. Skipping batch.")
                continue
            loss = loss_fn(logits, labels).item()
            prob = torch.softmax(logits, dim=1)
            y_true.append(labels.cpu())
            y_prob.append(prob.cpu())
            losses.append(loss)
    if not y_true:
        print("[WARN] No batches processed in evaluate.")
        return {}
    y_true = torch.cat(y_true)
    y_prob = torch.cat(y_prob)
    m = get_metrics(y_true, y_prob)
    m['loss'] = float(np.mean(losses)) if losses else 0.0
    return m


def train_fold(model, train_g, val_g, train_ids, val_ids, args, fold):
    device = args.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    sampler = MultiLayerNeighborSampler([int(x) for x in args.fanout.split(',')])

    train_loader = NodeDataLoader(
        train_g, {'document': torch.tensor(train_ids)}, sampler,
        batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = NodeDataLoader(
        val_g, {'document': torch.tensor(val_ids)}, sampler,
        batch_size=512, shuffle=False, num_workers=0
    )


    label_counts = torch.bincount(train_g.nodes['document'].data['label'])
    class_weights = (1.0 / label_counts.float()).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    history, best_f1, patience, best_metrics = defaultdict(list), 0., 0, None
    best_model_path = f"models/best_model_fold{fold}.pt"

    for epoch in range(1, args.epochs+1):
        model.train()
        for _, _, blocks in train_loader:
            blocks = [b.to(device) for b in blocks]
            feats  = {nt: blocks[0].srcnodes[nt].data['h'] for nt in blocks[0].srctypes}
            labels = blocks[-1].dstnodes['document'].data['label']
            logits = model(blocks, feats)
            if logits.size(0) != labels.size(0):
                print(f"[ERROR] Mismatch: logits={logits.size(0)} labels={labels.size(0)}. Skipping batch.")
                continue
            loss = loss_fn(logits, labels)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()

        scheduler.step()


        train_m = evaluate(model, train_g, train_ids, sampler, device, loss_fn)
        val_m   = evaluate(model, val_g, val_ids, sampler, device, loss_fn)

        for k in ['loss','acc','precision','recall','f1','auc']:
            history[f'train_{k}'].append(train_m.get(k, 0))
            history[f'val_{k}'].append(val_m.get(k, 0))

        print(f"Ep{epoch:02d} TrainLoss={train_m['loss']:.4f} ValLoss={val_m['loss']:.4f} "
              f"ACC={val_m['acc']:.3f} Prec={val_m['precision']:.3f} Rec={val_m['recall']:.3f} "
              f"F1={val_m['f1']:.3f} AUC={val_m['auc']:.3f}")

        if val_m['f1'] > best_f1:
            best_f1, patience, best_metrics = val_m['f1'], 0, val_m
            save_model(model, best_model_path)
        else:
            patience += 1
            if patience >= args.early_patience:
                print(f"Early stop fold{fold} at epoch {epoch}")
                break

    print("\nConfusion Matrix:\n", best_metrics['cm'])
    print("\nClassification Report:\n", best_metrics['report'])

    for k in ['loss','acc','precision','recall','f1','auc']:
        save_metric_plot(history[f'train_{k}'], history[f'val_{k}'], k.upper(), f'plots/{k}_fold{fold}.png')
    pd.DataFrame(history).to_csv(f'plots/metrics_fold{fold}.csv', index_label='epoch')
    return best_metrics


def main(args):
    ensure_dir('plots'); ensure_dir('models')

    df = pd.read_csv(args.train_tsv, sep='\t', dtype=str)
    codes = sorted(df['cpc_subclass'].unique())
    label_map = {c: i for i, c in enumerate(codes)}
    df['label'] = df['cpc_subclass'].map(label_map)
    global len_label; len_label = len(codes)

    kv = load_sgns(args.vector_path)
    embedder = StaticDeltaEmbedding(kv)
    encoder = CNNAttentionEncoder(kv.vector_size, channels=args.cnn_channels)

    skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=42)
    all_metrics = defaultdict(list)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label']), 1):
        print(f"\n===== Fold {fold}/{args.k_fold} =====")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)


        builder = PatentGraphBuilder(embedder, encoder)
        train_g, train_doc_nids = builder.build(train_df)

        builder = PatentGraphBuilder(embedder, encoder)
        val_g, val_doc_nids = builder.build(val_df)

        model = HeteroGATv2(train_g, encoder.out_dim, args.hidden_dim, args.num_heads, args.num_layers)
        m = train_fold(model, train_g, val_g, train_doc_nids, val_doc_nids, args, fold)

        for k in ['loss', 'acc', 'precision', 'recall', 'f1', 'auc']:
            all_metrics[k].append(m.get(k, 0))

    print("\n===== Cross-Validation Summary (mean ± std) =====")
    rows = []
    for k in ['loss', 'acc', 'precision', 'recall', 'f1', 'auc']:
        arr = np.array(all_metrics[k], dtype=float)
        mean = arr.mean() if arr.size > 0 else 0.0
        std  = arr.std(ddof=0) if arr.size > 0 else 0.0
        print(f"{k.upper():<10}: {mean:.4f} ± {std:.4f}")
        rows.append({'metric': k, 'mean': mean, 'std': std})


    pd.DataFrame(rows).to_csv("cv_summary.csv", index=False)
    print("Saved summary to cv_summary.csv")


if __name__ == "__main__":
    import dgl
    import torch
    import pandas as pd


    df = pd.read_csv("g_cpc_title_10000_filtered.tsv", sep='\t', dtype=str)
    codes = sorted(df['cpc_subclass'].unique())
    label_map = {c: i for i, c in enumerate(codes)}
    df['label'] = df['cpc_subclass'].map(label_map)

    kv = load_sgns("D:/CNNATTGAT/Public-data/numberbatch-en-19.08-plain.txt")
    embedder = StaticDeltaEmbedding(kv)
    encoder = CNNAttentionEncoder(kv.vector_size, channels=250)
    builder = PatentGraphBuilder(embedder, encoder)


    g, doc_ids = builder.build(df)


    for ntype in g.ntypes:

        if 'feat' in g.nodes[ntype].data:
            g.nodes[ntype].data['feat'] = g.nodes[ntype].data['feat'].float()
            continue
        if 'h' in g.nodes[ntype].data:
            g.nodes[ntype].data['feat'] = g.nodes[ntype].data['h'].detach().clone().float()
        else:
            raise RuntimeError(f"[{ntype}] Missing 'h' feature, unable to create a unified 'feat'. Please check the composition process.")

    for ntype in g.ntypes:
        ft = g.nodes[ntype].data['feat']
        print(f"[{ntype}] feat shape={tuple(ft.shape)}, dtype={ft.dtype}")

    dgl.save_graphs("graph_with_feat.bin", [g])
    print("✅ The image has been successfully saved as graph_with_feat.bin")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_tsv',      required=True, help="Path to g_cpc_title.tsv")
    parser.add_argument('--vector_path',    required=True, help="SGNS word vectors (word2vec format)")
    parser.add_argument('--epochs',         type=int, default=50)
    parser.add_argument('--early_patience', type=int, default=10)
    parser.add_argument('--batch_size',     type=int, default=128)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--k_fold',         type=int, default=5)
    parser.add_argument('--fanout',         type=str, default='5,5')
    parser.add_argument('--hidden_dim',     type=int, default=128)
    parser.add_argument('--num_heads',      type=int, default=4)
    parser.add_argument('--num_layers',     type=int, default=2)
    parser.add_argument('--cnn_channels',   type=int, default=250)
    parser.add_argument('--device',         type=str, default='cuda')
    args = parser.parse_args()
    torch.manual_seed(42)
    main(args)
