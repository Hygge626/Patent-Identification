import matplotlib
matplotlib.use('Agg')

import argparse
import os
from collections import defaultdict
from typing import List, Dict
import numpy as np
import pandas as pd
import jieba
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from rank_bm25 import BM25Okapi

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn import GATv2Conv, HeteroGraphConv
try:
    from dgl.dataloading import NodeDataLoader
except ImportError:
    from dgl.dataloading import DataLoader as NodeDataLoader
from dgl.dataloading import MultiLayerNeighborSampler


STOPWORDS = set()
try:
    with open('hit_stopwords.txt', encoding='utf-8') as f:
        STOPWORDS = set(line.strip() for line in f if line.strip())
    print(f"[INFO] Loaded {len(STOPWORDS)} stopwords")
except FileNotFoundError:
    print("[WARN] stopwords.txt not found, proceeding without stopword filtering.")


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
        safe_token = self._safe_key(token)
        if safe_token not in self.delta:
            self.delta[safe_token] = nn.Parameter(torch.zeros(self.dim))
        return base + self.delta[safe_token]


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
        """
        embeds: Tensor [L, emb_dim]
        returns: Tensor [out_dim]
        """
        L, D = embeds.shape

        if L <= self.position_encoding.size(1):
            pe = self.position_encoding[:, :L, :]
        else:
            extra = torch.zeros(1, L - self.position_encoding.size(1), D,
                                device=self.position_encoding.device)
            pe = torch.cat([self.position_encoding, extra], dim=1)

        x = embeds.unsqueeze(0) + pe

        x = x.transpose(1, 2)

        convs = [F.relu(conv(x)) for conv in self.convs]
        pools = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in convs]

        seq = torch.stack(pools, dim=1)

        attn_out, _ = self.attention(seq, seq, seq)
        out = attn_out.mean(1)
        return out.squeeze(0)


class PatentGraphBuilder:
    WINDOW = 5
    def __init__(self, embedder: StaticDeltaEmbedding, encoder: CNNAttentionEncoder):
        self.embedder = embedder
        self.encoder = encoder
        self.node_feats = defaultdict(list)
        self.node_offsets = defaultdict(int)
        self.word2nid: Dict[str,int] = {}
        self.edges = defaultdict(list)
        self.doc_nids, self.doc_labels = [], []

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

    @staticmethod
    def _split_sent(text: str) -> List[str]:
        dels='。！？!?'
        buf, out = '', []
        for ch in str(text):
            buf += ch
            if ch in dels:
                out.append(buf); buf = ''
        if buf:
            out.append(buf)
        return out

    def build(self, df: pd.DataFrame):
        abstracts = df['Abstract'].fillna('').astype(str).tolist()
        corpus_tokens = [list(jieba.cut(text)) for text in abstracts]
        tfidf_vec = TfidfVectorizer(max_features=5000)
        tfidf_matrix = tfidf_vec.fit_transform(abstracts)
        bm25 = BM25Okapi(corpus_tokens)
        bm25_scores = [bm25.get_scores(tokens)[idx] for idx, tokens in enumerate(corpus_tokens)]

        eff_cache = []
        for i, (_, row) in enumerate(df.iterrows()):
            label = int(row['Shared Value Proposition'])
            doc_id = self._add_node('document', torch.zeros(self.encoder.out_dim))
            self.doc_nids.append(doc_id); self.doc_labels.append(label)

            # title
            title_toks = [t for t in jieba.cut(str(row['Title'])) if t not in STOPWORDS]
            title_id = self._add_node('title', self._encode(title_toks))
            self.edges[('document','has_title','title')].append((doc_id,title_id))

            # claim
            claim_toks = [t for t in jieba.cut(str(row['Independent claim'])) if t not in STOPWORDS]
            claim_id = self._add_node('claim', self._encode(claim_toks))
            self.edges[('document','has_claim','claim')].append((doc_id,claim_id))

            # efficacy
            eff_toks = [t for t in jieba.cut(str(row['Technical efficacy sentence'])) if t not in STOPWORDS]
            eff_id = self._add_node('efficacy', self._encode(eff_toks))
            self.edges[('document','has_efficacy','efficacy')].append((doc_id,eff_id))
            eff_cache.append((eff_id, str(row['Technical efficacy sentence'])))

            # abstract -> sentence -> word
            for sent in self._split_sent(row['Abstrac']):
                if not sent.strip(): 
                    continue
                s_toks = [t for t in jieba.cut(sent) if t not in STOPWORDS]
                sent_id = self._add_node('sentence', self._encode(s_toks))
                for idx_tok, tok in enumerate(s_toks):
                    w1 = self._get_word(tok)
                    self.edges[('sentence','has_word','word')].append((sent_id,w1))
                    for j in range(idx_tok+1, min(idx_tok+self.WINDOW, len(s_toks))):
                        w2 = self._get_word(s_toks[j])
                        self.edges[('word','cooccur','word')].extend([(w1,w2),(w2,w1)])

            self.edges[('claim','has_word','word')] += [
                (claim_id, self._get_word(t)) for t in claim_toks
            ]
            self.edges[('efficacy','has_word','word')] += [
                (eff_id, self._get_word(t)) for t in eff_toks
            ]

            row_tfidf = tfidf_matrix.getrow(i)
            for idx, val in zip(row_tfidf.indices, row_tfidf.data):
                token = tfidf_vec.get_feature_names_out()[idx]
                if token in self.word2nid:
                    self.edges[('document','has_tfidf_word','word')].append((doc_id,self.word2nid[token]))

        ipc_map = defaultdict(list)
        for i, (_, row) in enumerate(df.iterrows()):
            ipc_codes = row['IPC'].split(';')
            for code in ipc_codes:
                ipc_map[code.strip()].append(i)
        for ipc_code, doc_ids in ipc_map.items():
            for m in range(len(doc_ids)):
                for n in range(m+1, len(doc_ids)):
                    self.edges[('document','ipc_sim','document')].append((doc_ids[m],doc_ids[n]))


        cos_mat = cosine_similarity(tfidf_matrix)
        for m in range(cos_mat.shape[0]):
            for n in range(m+1, cos_mat.shape[1]):
                if cos_mat[m,n] > 0.6:
                    self.edges[('document','doc_sim','document')].extend([(m,n),(n,m)])


        rev_edges = {}
        for (src, etype, dst), pairs in list(self.edges.items()):
            rev_edges[(dst,f"rev_{etype}",src)] = [(d,s) for s,d in pairs]
        for k, v in rev_edges.items():
            self.edges[k] += v


        g_data = {
            etype: (torch.tensor([s for s,_ in pairs]), torch.tensor([d for _,d in pairs]))
            for etype, pairs in self.edges.items() if pairs
        }
        g = dgl.heterograph(g_data)


        for ntype, feats in self.node_feats.items():
            g.nodes[ntype].data['h'] = torch.stack(feats).detach()

        g.nodes['document'].data['label'] = torch.tensor(self.doc_labels)
        g.nodes['document'].data['tfidf'] = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float)
        g.nodes['document'].data['bm25'] = torch.tensor(bm25_scores, dtype=torch.float).unsqueeze(1)

        return g, self.doc_nids


class HeteroGATv2(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, in_dim, hidden_dim, heads, layers):
        super().__init__()
        self.feat_project = nn.ModuleDict({
            ntype: nn.Linear(300 if ntype=='word' else in_dim, in_dim)
            for ntype in graph.ntypes
        })
        self.layers = nn.ModuleList()
        for l in range(layers):
            convs = {}
            for src, et, dst in graph.canonical_etypes:
                convs[(src,et,dst)] = GATv2Conv(
                    in_dim if l==0 else hidden_dim*heads,
                    hidden_dim, heads,
                    feat_drop=0.2, attn_drop=0.2,
                    allow_zero_in_degree=True
                )
            self.layers.append(HeteroGraphConv(convs, aggregate='mean'))
        self.clf = nn.Sequential(
            nn.Linear(hidden_dim*heads, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, blocks, h_dict):
        h = {nt: self.feat_project[nt](feat) for nt, feat in h_dict.items()}
        for layer, blk in zip(self.layers, blocks):
            out = layer(blk, h)
            h = {k: v.flatten(1) for k, v in out.items()}
        if 'document' not in h or h['document'].size(0) == 0:
            return torch.zeros((0,2), device=next(self.parameters()).device)
        logits = self.clf(h['document'])
        return logits


def get_metrics(y_true: torch.Tensor, y_prob: torch.Tensor) -> Dict:
    y_pred = (y_prob >= 0.5).long()
    return {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'cm': confusion_matrix(y_true, y_pred),
        'report': classification_report(y_true, y_pred, digits=3)
    }

def evaluate(model, g, nids, sampler, device):
    model.eval()
    loader = NodeDataLoader(g, {'document': torch.tensor(nids)}, sampler,
                            batch_size=512, shuffle=False, num_workers=0)
    y_true, y_prob, losses, batch_cnt = [], [], [], 0
    with torch.no_grad():
        for _, _, blocks in loader:
            blocks = [b.to(device) for b in blocks]
            feats = {nt: blocks[0].srcnodes[nt].data['h'].to(device) for nt in blocks[0].srctypes}
            labels = blocks[-1].dstnodes['document'].data['label'].to(device)
            logits = model(blocks, feats)
            if logits.size(0) == 0:
                continue
            prob = torch.softmax(logits, dim=1)[:,1]
            losses.append(F.cross_entropy(logits, labels).item())
            y_true.append(labels.cpu()); y_prob.append(prob.cpu()); batch_cnt += 1
    if batch_cnt == 0:
        print("[WARN] No valid document nodes in validation.")
        return {'acc':0,'precision':0,'recall':0,'f1':0,'auc':0,'loss':0,'cm':np.zeros((2,2)),'report':'No valid data.'}
    y_true = torch.cat(y_true); y_prob = torch.cat(y_prob)
    m = get_metrics(y_true, y_prob); m['loss'] = float(np.mean(losses))
    return m

def train_fold(model, g, train_ids, val_ids, args, fold):
    device = args.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    sampler = MultiLayerNeighborSampler([int(x) for x in args.fanout.split(',')])
    loader = NodeDataLoader(g, {'document': torch.tensor(train_ids)}, sampler,
                            batch_size=args.batch_size, shuffle=True, num_workers=0)

    history = defaultdict(list)
    best_f1, patience, best_metrics = 0., 0, {}
    best_model_path = f"models/best_model_fold{fold}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, steps = 0., 0
        for _, _, blocks in loader:
            blocks = [b.to(device) for b in blocks]
            feats = {nt: blocks[0].srcnodes[nt].data['h'].to(device) for nt in blocks[0].srctypes}
            labels = blocks[-1].dstnodes['document'].data['label'].to(device)
            logits = model(blocks, feats)
            if logits.size(0) == 0:
                continue
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            epoch_loss += loss.item(); steps += 1

        with torch.no_grad():
            model.eval()
            train_m = evaluate(model, g, train_ids, sampler, device)
            val_m = evaluate(model, g, val_ids, sampler, device)

        scheduler.step()

        for key in ['loss', 'acc', 'precision', 'recall', 'f1', 'auc']:
            history[f'train_{key}'].append(train_m[key])
            history[f'val_{key}'].append(val_m[key])

        print(f"Ep{epoch:02d} TrainLoss={train_m['loss']:.4f} ValLoss={val_m['loss']:.4f} "
              f"ACC={val_m['acc']:.3f} Prec={val_m['precision']:.3f} "
              f"Rec={val_m['recall']:.3f} F1={val_m['f1']:.3f} AUC={val_m['auc']:.3f}")

        if val_m['f1'] > best_f1:
            best_f1 = val_m['f1']; patience = 0; best_metrics = val_m
            save_model(model, best_model_path)
        else:
            patience += 1
            if patience >= args.early_patience:
                print(f"Early stop at epoch {epoch}")
                break

    print("\nConfusion Matrix:\n", best_metrics['cm'])
    print("\nClassification Report:\n", best_metrics['report'])


    for key in ['loss', 'acc', 'precision', 'recall', 'f1', 'auc']:
        save_metric_plot(history[f'train_{key}'], history[f'val_{key}'], key.upper(), f'plots/{key}_fold{fold}.png')


    pd.DataFrame(history).to_csv(f'plots/metrics_fold{fold}.csv', index_label='epoch')

    return best_metrics

def main(args):
    ensure_dir('plots'); ensure_dir('models')
    df = pd.read_excel(args.train_xlsx)[['Shared Value Proposition','IPC','Title','Abstract','Independent claim','Technical efficacy sentence']]
    kv = load_sgns(args.vector_path)
    embedder = StaticDeltaEmbedding(kv)
    encoder = CNNAttentionEncoder(kv.vector_size, channels=args.cnn_channels)
    builder = PatentGraphBuilder(embedder, encoder)
    g, doc_ids = builder.build(df)
    labels = g.nodes['document'].data['label'].numpy()
    skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=42)

    all_metrics = defaultdict(list)
    for fold, (train_idx, val_idx) in enumerate(skf.split(doc_ids, labels), 1):
        print(f"\n========== Fold {fold}/{args.k_fold} ========== ")
        model = HeteroGATv2(g, encoder.out_dim, args.hidden_dim, args.num_heads, args.num_layers)
        metrics = train_fold(model, g, train_idx.tolist(), val_idx.tolist(), args, fold)
        for k in ['acc','precision','recall','f1','auc']:
            all_metrics[k].append(metrics[k])
        print(f"Fold {fold} - ACC: {metrics['acc']:.4f}, Prec: {metrics['precision']:.4f}, "
              f"Rec: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")

    print("\n===== Cross-Validation Summary =====")
    for k in ['acc','precision','recall','f1','auc']:
        print(f"{k.upper()}: {np.mean(all_metrics[k]):.4f}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_xlsx', required=True)
    parser.add_argument('--vector_path', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--early_patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--fanout', type=str, default='5,5')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--cnn_channels', type=int, default=250)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    torch.manual_seed(42)
    main(args)
