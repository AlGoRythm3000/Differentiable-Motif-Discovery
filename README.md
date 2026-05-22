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

To ensure the sampling operation remains differentiable for backpropagation, we employ the **Gumbel-Softmax** continuous relaxation.

### 3. Permutation-invariant encoding
The sampled nodes form an unorderd set $\mathcal{S}$. To aggregate these nodes into a single, fixed-size motif representation $h_{\mathcal{S}}$, we utilize a **DeepSets** architecture. This ensures that the structural encoding is strictly invariant to node permutation within the discovered motif.

### 4. Differentiable scoring and rewiring
A Multi-Layer Perceptron (MLP) evaluates the aggregated motif representation $h_{\mathcal{S}}$ to output a continuous existence probability (weight). Motifs with negligible weights are pruned, dynamically rewiring the graph with soft, task-relevant higher-order connections.

### 5. End-to-end optimization <!--with Sparsity-->
A downstream topological message-passing layer operates on this rewired structure. The entire pipeline is optimized jointly using the downstream task loss (e.g., cross-entropy for node classification). To prevent structural density explosion (oversmoothing/computational bottleneck), a sparsity-promoting regularization term is added to the objective function, forcing the model to select only the most informative motifs.

## How to reproduce the results

```bash
python main.py
```
--- 

## Project structure
1. Create the conda environment : 
(automatically called "dmd")
```bash
conda env create --file=environment.yml
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
│   ├── embeddings.py         # different types of embeddings : GNNs, SPE, FoMLP, etc.  
│   ├── motif_generator.py  # 
│   ├── motifs.py           # motifs encoding
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

├── tools/ # ou "utils" ?? déjà un fichier "utils" ...
│   ├── __init__.py
│   ├── metrics.py          # accuracy, ROC, AUC, ...
│   └── losses.py           # task + sparsity loss

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

