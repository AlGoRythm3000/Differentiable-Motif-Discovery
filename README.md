# Differentiable Motif Discovery (DMD) for Topological Neural Networks
*This repository is the repo of the first part of my internship at INRIA / CentraleSupélec it contains the implementation of the DMD framework, a generative approach to topological deep learning.*

There is a link to the [paper](https://). 


<!-- ## Method and pipeline -->

<!-- **Step 1 :** Node Embeddings

 Un premier GNN (souvent simple, comme un GCN ou un GIN) est appliqué sur le graphe de départ pour obtenir les représentations latentes :$Z = \text{GNN}_{\text{base}}(X, A)$ où chaque ligne correspond à un vecteur $z_u$.

Étape 2 : Le Processus Génératif (Motif Proposal)C'est l'étape la plus audacieuse. Au lieu de chercher des triangles algorithmiquement, on utilise l'espace latent pour échantillonner des sous-graphes.Pour un nœud "graine" $v$, on calcule la probabilité de recruter n'importe quel autre nœud $u$ du graphe dans son motif :

$$p_\phi(u|v) = \text{Softmax}\left( \frac{\text{sim}(z_v, z_u)}{\tau} \right)$$

Où $\text{sim}$ est une fonction de similarité (produit scalaire, ou une forme bilinéaire paramétrée $\phi$) et $\tau$ est une température. Détail technique : Pour que ce tirage probabiliste reste différentiable vis-à-vis de la fonction de perte finale (qui viendra bien plus tard), on utilise généralement l'astuce de Gumbel-Softmax. Cela crée un ensemble flou de nœuds $\mathcal{S} = \{v, u_1, u_2, \dots\}$.


Étape 3 : L'Encodage Invariant par Permutation (Focus sur DeepSets)Une fois l'ensemble $\mathcal{S}$ échantillonné, il faut le transformer en un vecteur unique $h_{\mathcal{S}}$ (le "plongement du motif").Le problème : Un motif est un ensemble (un Set). Mathématiquement, l'ensemble $\{u_1, u_2\}$ est strictement identique à l'ensemble $\{u_2, u_1\}$. Si tu utilises un réseau classique (MLP) concaténant les vecteurs, ou un RNN, la sortie changera selon l'ordre dans lequel tu présentes les nœuds. C'est un non-sens topologique.La solution (DeepSets) : Introduit par Zaheer et al. (2017), le théorème de DeepSets prouve que pour qu'une fonction opérant sur un ensemble soit universelle et invariante par permutation, elle doit prendre la forme suivante :$$h_{\mathcal{S}} = \rho \left( \bigoplus_{u \in \mathcal{S}} \phi(z_u) \right)$$$\phi$ (souvent un MLP) est appliqué à chaque nœud indépendamment pour en extraire les caractéristiques utiles à la formation du motif.$\bigoplus$ est un opérateur d'agrégation symétrique (la somme, la moyenne, ou le max). L'ordre des éléments n'a plus aucune importance. La somme est généralement privilégiée car elle préserve l'information sur la taille (la cardinalité) du motif.$\rho$ (un autre MLP) prend cette somme et produit l'embedding final du motif d'ordre supérieur.

Étape 4 : Évaluation (Scoring & Rewiring)Le vecteur du motif $h_{\mathcal{S}}$ est passé dans un dernier classifieur (un petit MLP avec une activation Sigmoïde) pour obtenir un poids $w_{\mathcal{S}} \in [0, 1]$. C'est la probabilité que ce motif généré soit réellement utile pour le graphe. Les motifs avec un poids proche de 0 sont ignorés (rewiring dynamique).

Étape 5 : Optimisation et Parcimonie (Sparsity Loss)On fait passer le GNN final sur cette nouvelle topologie enrichie par les motifs découverts. La fonction de perte (Loss) de la tâche principale (ex: classification) dicte la direction du gradient, qui remonte jusqu'à l'étape 2 pour ajuster la manière de générer les motifs.Le défi technique : Si on ne fait rien, le réseau va trouver que tout lier à tout (un graphe complet) maximise temporairement le flux d'information. C'est catastrophique (complexité computationnelle explosive et Oversmoothing massif). Il est donc indispensable d'ajouter un terme de régularisation dans la loss :

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{sparse}}$$

--- -->

*This repository contains the implementation of the DMD framework, a generative approach to topological deep learning.*

## Methodology : the DMD pipeline

Unlike standard Topological Neural Networks (TNNs) that rely on predefined deterministic algorithms to lift graphs (e.g., extracting cycles or cliques), our framework infers latent higher-order structures directly from the data in an end-to-end differentiable manner. 

The pipeline transitions from **Topological Selection** to **Latent Topology Inference** through the following 5 key steps:

### 1. Base latent representation
A base Graph Neural Network (e.g., GCN, GIN) processes the initial graph $\mathcal{G}=(\mathcal{V},\mathcal{E},X)$ to map nodes into a continuous latent space, producing node embeddings $Z \in \mathbb{R}^{|\mathcal{V}| \times d}$. 

### 2. Stochastic motif proposal
For a given target node $v$, a subset of candidate nodes $\mathcal{S}$ is sampled to form a higher-order motif. Instead of rigid geometric constraints, the sampling probability is parameterized by the pairwise similarity in the latent space :

$$p_\phi(u | v) \propto \exp(\text{sim}(z_v, z_u))$$

By default (the `topk` brick) the members of a candidate cell are selected by a hard top-$k$ over these similarities. That selection carries **no gradient**: only the aggregated similarity score of a cell does, so the model learns *how much to trust* a cell rather than *whom to recruit*. The continuous relaxation actually applied by `topk` (Gumbel-**sigmoid**, with a straight-through estimator) happens one stage later, at the accept/reject decision of step 4 — not here, despite the name suggesting otherwise. Membership itself only becomes differentiable under the `autoregressive` brick (`feat/rich-bricks`), which builds a cell member-by-member, each step a genuine **Gumbel-softmax** draw over the remaining candidates conditioned on the running set; see [Rich bricks](#rich-bricks) below.

### 3. Permutation-invariant encoding
The sampled nodes form an unorderd set $\mathcal{S}$. To aggregate these nodes into a single, fixed-size motif representation $h_{\mathcal{S}}$, we utilize a **DeepSets** architecture. This ensures that the structural encoding is strictly invariant to node permutation within the discovered motif.

### 4. Differentiable scoring and rewiring
Each candidate cell's score is turned into a continuous acceptance weight $\alpha \in [0,1]$ by a **Gumbel-sigmoid** relaxation (a relaxed Bernoulli), optionally hardened by a straight-through estimator so the forward value is binary while the gradient still flows through the soft weight. Cells with negligible weight contribute nothing, dynamically rewiring the graph with soft, task-relevant higher-order connections.

### 5. End-to-end optimization <!--with Sparsity-->
A downstream message-passing layer operates on this rewired structure, and the whole pipeline is optimized jointly:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \mu\,\mathcal{L}_{\text{sparsity}} + \gamma\,\mathcal{L}_{\text{OSq}}$$

The sparsity term prevents structural density explosion (the objective would otherwise be minimized by connecting everything to everything). The third term scores the **oversquashing** of the rewired structure and is the object of the current work; it is evaluated on the *soft* acceptance weights, never on a thresholded structure, which is what makes it differentiable. See [OSq objective](#oversquashing-objective) below.

## Rich bricks

`feat/rich-bricks` makes every stage config-driven: each of the five stages
above is selectable by a string (`models/registry.py`), so a "simple" and a
"rich" implementation are always a drop-in swap for each other. Assembled by
`models/dmd_model.py::DMDModel`, e.g. `DMDModel(..., s1="gpse", s2="cycle_basis",
s5="tnn")`, or via `train.py`'s `--encoder/--proposal/--cell-encoder/--selector-type/--message-passing`
flags, or via a declarative config under `configs/` (see below).

| stage | simple (default) | rich bricks |
|---|---|---|
| 1. node embeddings | `gcn`, `gin` | `gpse` - frozen pretrained structural encoder (Cantürk et al. 2024), cached per dataset (`tools/gpse_cache.py`); falls back to `pse_explicit` if the checkpoint can't be fetched |
| | | `pse_explicit` - LapPE + RWSE via PyG transforms |
| 2. candidate proposal | `topk` | `autoregressive` - builds each cell member-by-member, Gumbel-softmax per step; the only brick with differentiable membership |
| | | `cycle_basis` - DiffLift-style: enumerates graph cycles (Paton's algorithm via `networkx.cycle_basis`), fixed membership |
| 3. cell encoder | `deepsets` | `set_transformer` - ISAB + PMA attention over the cell |
| | | `mini_gnn` - a small GNN over the cell's induced subgraph; the only encoder that tells a triangle from a 3-path |
| 4. selection | `gumbel` | `ksubset` - relaxed top-k with explicit cardinality control |
| | | `reinforce` - score-function estimator with an EMA baseline; the honest high-variance comparison point |
| 5. message passing | `gnn_rewired` | `tnn` - a genuine cell complex, message-passed with TopoModelX's CWN layer (Bodnar et al. 2021); requires `s2=cycle_basis` |
| | | `hypergraph_tnn` - accepted cells as hyperedges, message-passed with TopoModelX's UniGCN layer (Huang & Yang 2021); works with any proposal |

`models/registry.py::validate_pipeline_config` rejects invalid combinations
(currently: `s5=tnn` without `s2=cycle_basis`) with a clear error rather than
silently building an invalid complex.

## Oversquashing objective

Oversquashing is *measured* and *optimized* by two deliberately separate modules, so that
"we optimize X and the measured X moves" stays an auditable claim:

| module | when | what |
|---|---|---|
| `tools/osq_metrics.py` | analysis time | exact effective resistance, $\lambda_2$, edge curvature, betweenness-weighted curvature (`wc`/`nwc`), influence decay. Reported, never optimized. |
| `tools/osq_proxies.py` | training time | differentiable, cheap surrogates, swappable by name. Optimized, never reported as the measurement. |

Proxies available (`--osq-proxy`):

| key | what | role |
|---|---|---|
| `none` | exact zero | baseline arm |
| `r_bar` | mean effective resistance, Hutchinson probes + conjugate gradient, closed-form backward | **primary** |
| `lambda2` | spectral-gap surrogate via deflated power iteration + Rayleigh quotient | guard-rail |
| `efc` | smooth penalty on negatively curved edges, paired against the original 1-skeleton | cheap, local |
| `cf_bc_efc` | curvature weighted by current-flow betweenness, reusing `r_bar`'s solves | local + global |

`r_bar` never forms $L^{+}$: it estimates $\mathrm{tr}(L^{+})$ from $k$ random probes solved by
CG ($O(|E|)$ per iteration), and its gradient
$\partial\,\mathrm{tr}(L^{+})/\partial w_e = -\lVert L^{+} b_e\rVert^2$ is read off the very
solutions the forward pass already computed — so the backward pass is free, and it is local,
which is what lets it chain back to each cell's acceptance weight. The gradient is checked
against finite differences, and against the qualitative requirement that on two cliques
joined by a bridge, the bridge is the edge it pushes hardest on.

Two rules the code enforces and the tests pin down:

* the proxy scores the **soft** structure (acceptance weights as continuous edge weights),
  never a rounded one;
* an OSq term is **never** run without the sparsity counterweight — every proxy here is
  minimized by the complete graph.

```bash
# proxy-free run (unchanged behaviour)
python train.py --task graph_classification --dataset MUTAG

# with the primary proxy, and the before/after measurement stored in summary.json
python train.py --task graph_classification --dataset MUTAG \
    --osq-proxy r_bar --osq-weight 0.1 --sparsity-weight 0.05 --osq-report
```

`--osq-weight 0` short-circuits the proxy entirely, so it reproduces the proxy-free
objective bit for bit whatever `--osq-proxy` says.

### Experiment grid

`tools/experiment_grid.py` runs the Tier A/B/C pipeline configs under `configs/`
(one brick swapped at a time, then crossed - see [Rich bricks](#rich-bricks) and
`configs/`'s own files for the exact list) × `gamma ∈ {0, best}` × datasets × seeds,
writes an append-only results tree (`runs.csv`, `epochs.csv`, `env.json`,
`raw/<run_id>.json`, schema frozen in `tools/results_store.py`), records a commit SHA
on every row, never overwrites an existing `run_id`, and stores the OSq measurement
**before and after** the lifting for each run. `run_id` is
`f"{tier}_{config_id}_{dataset}_g{gamma}_s{seed}"`.
`notebooks/kaggle_rich_bricks.ipynb` is the self-contained, sharded Kaggle front-end: it
clones a pinned branch, precomputes/caches GPSE encodings for whichever datasets need them,
runs one shard of the grid, and zips the results.

The grid includes a synthetic arm from `tools/synthetic.py` (bottleneck graph families with
a query/answer matching task that cannot be solved without pushing information across the
bottleneck). That arm is the falsification core — it is where an OSq-guided lifting is
supposed to win — and the runtime guard never drops it.

## How to reproduce the results

```bash
python main.py
```

The pretrained GPSE weights are not tracked by git (~254 MiB). Fetch them once with
`./download_gpse.sh` — see [GPSE_pretrained/README.md](GPSE_pretrained/README.md).
--- 

## Project structure
1. Create and activate the conda environment : 
(automatically called "dmd")
```bash
conda env create --file=environment.yml
conda activate dmd
```

```bash
Differentiable-Motif-Discovery/
├── datasets/       
├── layers/                 # folder containing the layers gnn, tnn => can also be taken from topomodelX
│   ├── 
│   ├── 
│   └──             
├── models/                 # the general architecture
│   ├── __init__.py
│   ├── graph_embeddings.py         # different types of embeddings : GNNs, SPE, FoMLP, etc.  
│   ├── motif_proposition.py  # 
│   ├── motifs_encodings.py           # motifs encoding
│   ├── weight_assignement.py
│   └── dmd_model.py        # 
├── resources/              # folder containing the images, or any other media
│   ├── 
│   ├── 
│   └── 
├── tasks/                  # task to be run
│   ├── node_classification.py
│   └── graph_classification.py

├── tests/                  # directory for all the test functions
│   ├── test_main.py
│   └── test_train.py

├── tools/
│   ├── __init__.py
│   ├── metrics.py          # accuracy, ROC, AUC, ...
│   ├── losses.py           # task + sparsity + oversquashing loss
│   ├── osq_metrics.py      # OSq measurement (exact, analysis time)
│   ├── osq_proxies.py      # OSq proxies (differentiable, training time)
│   ├── synthetic.py        # bottleneck graph families (falsification core)
│   ├── experiment_grid.py  # datasets x proxies x gamma x seeds
│   ├── results_store.py    # append-only results schema
│   └── experiment_logger.py

├── results/ 
│   ├── analyze_results.py
│   └── 
├── train.py                # main training loop (forward, backward, optim)
├── evaluate.py             # test script
├── environment.yml         # file containing the used packages   
└── main.py                 # Le point d'entrée exécutable (avec le parsing d'arguments)

# ├── nom_dossier/ 
# │   ├── 
# │   └──


# besoin d'un comparaison avec difflift et DCM 
```

