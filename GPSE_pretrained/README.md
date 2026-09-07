# Pretrained GPSE weights

This folder holds the pretrained **GPSE** (Graph Positional and Structural Encoder)
checkpoints. The `.pt` files are ~254 MiB each, so they are **not tracked by git** —
download them once after cloning.

## Quick start

From the repository root:

```bash
./download_gpse.sh
```

That fetches `gpse_model_molpcba_1.0.pt` (the variant used in this repo), resumes an
interrupted download, skips the file if it is already there, and verifies its SHA-256.

Other variants: `./download_gpse.sh zinc | pcqm4mv2 | geom | chembl | all`.

## Manual download

If you would rather not run the script:

```bash
# with curl
curl -L -o GPSE_pretrained/gpse_model_molpcba_1.0.pt \
  "https://zenodo.org/records/8145095/files/gpse_model_molpcba_1.0.pt?download=1"

# or with wget
wget -O GPSE_pretrained/gpse_model_molpcba_1.0.pt \
  "https://zenodo.org/records/8145095/files/gpse_model_molpcba_1.0.pt?download=1"
```

Expected file:

| file | size (bytes) | sha256 |
| --- | --- | --- |
| `gpse_model_molpcba_1.0.pt` | 266238730 | `7bc81da2231cc1a6898683317fe0a278ffa4953540bbad375e2c20af840f9cdd` |

Check it with:

```bash
sha256sum GPSE_pretrained/gpse_model_molpcba_1.0.pt
```

## Source

Zenodo record [8145095](https://zenodo.org/records/8145095) — "Graph Positional and
Structural Encodings (model weights and precomputed encodings)", released with
[*Graph Positional and Structural Encoder*](https://arxiv.org/abs/2307.07107)
(Cantürk et al., ICML 2024). Code: <https://github.com/G-Taxonomy-Workgroup/GPSE>.

The same record also serves `zinc`, `pcqm4mv2`, `geom` and `chembl` checkpoints
(same 266238730-byte size, different weights).

PyG ships a GPSE implementation as well, which can pull the weights on its own:

```python
from torch_geometric.nn.models import GPSE
model = GPSE.from_pretrained("molpcba", root="GPSE_pretrained")
```
