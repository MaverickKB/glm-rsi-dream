# External runtime terms

The source code in this repository uses MIT and Apache-2.0 licenses. Downloaded checkpoints and containers have their own terms. They are separate prerequisites, and this package redistributes none of their weights or binaries.

The pinned TR3 checkpoint is `brandonmusic/GLM-5.3-Flash-tr3-4bpw`. The archived packaging evidence identifies its ShapleyMcg License v1.0, including required attribution and a named exclusion. The attribution is retained in NOTICE. This is a separate model-use condition, not an MIT or Apache license for the checkpoint.

The pinned DFlash2 drafter is `incoai/GLM-5.3-Flash-DFlash2`. The retained upstream notice records CC BY-NC-ND 4.0 terms: non-commercial use, no derivatives. The fresh-pair provisioning recipe downloads and configures this drafter. Learning/evaluation requests disable draft decoding; that request setting does not change the downloaded model's terms.

The external NVIDIA-based runtime image includes software governed by NVIDIA's container terms and its other bundled licenses. Retain and review the downloaded notices. Source pins identify compatibility; they confer no additional license rights.
