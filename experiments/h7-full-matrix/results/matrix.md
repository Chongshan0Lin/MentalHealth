
### `mmpsy_depression` — structural shortcut (answer-segment count)

Shortcut rate (model-free): **segment count 0.772 (content 0.507, chance 0.500)**

| model | n | CSS | 95% CI | flips | native AUC | native acc | worst-arm acc | verdict |
|---|---|---|---|---|---|---|---|---|
| MentaLLaMA | 190 | **-0.0006** | [-0.0023, +0.0011] | 0.037 | 0.463 | 0.274 | 0.274 | **null** |
| Menta | 190 | **+0.0017** | [-0.0084, +0.0107] | 0.037 | 0.790 | 0.353 | 0.332 | **null** |
| qwen3_base | 190 | **-0.0000** | [-0.0000, +0.0000] | 0.000 | 0.642 | 0.816 | 0.816 | **null** |
| OSim-8B | 190 | **-0.0006** | [-0.0009, -0.0003] | 0.000 | 0.768 | 0.816 | 0.816 | **null** |
| OmniSapiens-SFT | 190 | **+0.0046** | [-0.0002, +0.0101] | 0.016 | 0.883 | 0.711 | 0.711 | **null** |
| OmniSapiens-RL | 190 | **+0.0041** | [+0.0015, +0.0066] | 0.011 | 0.871 | 0.847 | 0.821 | **null** |
| OmniSapiens2.0 | 190 | **+0.0011** | [-0.0036, +0.0056] | 0.011 | 0.885 | 0.863 | 0.853 | **null** |
| HumanOmniV2 | 190 | **+0.0028** | [+0.0007, +0.0048] | 0.005 | 0.783 | 0.821 | 0.816 | **null** |

(8/8 models)

### `ptsd_in_the_wild` — register shortcut, cross-source

Shortcut rate (model-free): **text 0.948, stopwords only 0.880 (chance 0.500)**

| model | n | CSS | 95% CI | flips | native AUC | native acc | worst-arm acc | verdict |
|---|---|---|---|---|---|---|---|---|
| MentaLLaMA | 60 | **-0.0021** | [-0.0051, +0.0007] | 0.000 | 0.473 | 0.500 | 0.500 | **null** |
| Menta | — | — | — | — | — | — | — | *not run* |
| qwen3_base | 60 | **+0.0197** | [+0.0023, +0.0466] | 0.017 | 0.993 | 0.967 | 0.967 | **null** |
| OSim-8B | 60 | **+0.0164** | [+0.0105, +0.0221] | 0.000 | 0.980 | 0.500 | 0.500 | **null** |
| OmniSapiens-SFT | 60 | **+0.1920** | [+0.1433, +0.2421] | 0.433 | 0.997 | 0.967 | 0.567 | moderate |
| OmniSapiens-RL | 60 | **+0.0960** | [+0.0778, +0.1146] | 0.167 | 1.000 | 0.700 | 0.500 | weak |
| OmniSapiens2.0 | 60 | **+0.2824** | [+0.2144, +0.3484] | 0.483 | 1.000 | 1.000 | 0.517 | **strong** |
| HumanOmniV2 | 60 | **+0.0227** | [+0.0160, +0.0299] | 0.000 | 0.973 | 0.500 | 0.500 | weak |

(7/8 models)

### `Irf` — register shortcut, same-source

Shortcut rate (model-free): **text 0.643, stopwords only 0.622 (chance 0.500)**

| model | n | CSS | 95% CI | flips | native AUC | native acc | worst-arm acc | verdict |
|---|---|---|---|---|---|---|---|---|
| MentaLLaMA | 400 | **-0.0822** | [-0.0954, -0.0695] | 0.077 | 0.849 | 0.757 | 0.752 | **wrong sign** |
| Menta | 400 | **-0.0239** | [-0.0322, -0.0164] | 0.015 | 0.673 | 0.685 | 0.670 | **wrong sign** |
| qwen3_base | 400 | **-0.0073** | [-0.0209, +0.0048] | 0.022 | 0.716 | 0.713 | 0.675 | **null** |
| OSim-8B | 400 | **+0.0238** | [+0.0173, +0.0308] | 0.015 | 0.792 | 0.745 | 0.672 | weak |
| OmniSapiens-SFT | 400 | **-0.0010** | [-0.0014, -0.0006] | 0.000 | 0.761 | 0.682 | 0.667 | **null** |
| OmniSapiens-RL | 400 | **-0.0119** | [-0.0132, -0.0106] | 0.000 | 0.755 | 0.677 | 0.667 | **null** |
| OmniSapiens2.0 | 400 | **-0.0078** | [-0.0089, -0.0067] | 0.000 | 0.745 | 0.685 | 0.667 | **null** |
| HumanOmniV2 | 400 | **-0.0462** | [-0.0533, -0.0392] | 0.025 | 0.753 | 0.718 | 0.660 | **wrong sign** |

(8/8 models)
