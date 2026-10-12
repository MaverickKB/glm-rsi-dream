# Public dependencies and provenance

The hardware target is two NVIDIA DGX Spark GB10 nodes with TP=2. The head owns
the coordinator and generation API; the worker remains rank 1. Host addresses,
SSH identity, storage paths, ports, fabric interfaces, and protected probes are
operator-owned configuration.

## Borrowed source and notices

| Material | Public source | License | Retained attribution |
| --- | --- | --- | --- |
| TensorFold native source and overlay context | [TensorFold 0.6.0, commit c4646171139ee8a3c38103eaa1699dad226ec12b](https://github.com/ashhart/TensorFold/tree/c4646171139ee8a3c38103eaa1699dad226ec12b) | Apache-2.0; pre-0.6.0 code retains MIT notice | TensorFold Apache, legacy MIT, and NOTICE files under licenses/ |
| Two-Spark TensorFold integration and modified source fixtures | [MiaAI-Lab TensorFold recipe, reviewed commit 3fd60dfc836fab0f1ed0fd4f0afaf2a1d62b2aab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/3fd60dfc836fab0f1ed0fd4f0afaf2a1d62b2aab) | Apache-2.0 with retained upstream MIT and Apache notices | Mia-stack-LICENSE.txt and Mia-stack-NOTICE.txt |
| Contributor Covenant 2.1, CODE_OF_CONDUCT.md | [Historical 2.1 text and license, commit b58a318f59db5dd8e1a32119339ef1921da3d3e1](https://github.com/EthicalSource/contributor_covenant/tree/b58a318f59db5dd8e1a32119339ef1921da3d3e1) | CC BY 4.0, documentation exception | Code-of-conduct attribution and Contributor-Covenant-CC-BY-4.0.txt |

The two-Spark dependency is the repository whose name ends in **TensorFold**.
Its reviewed public LICENSE and NOTICE match the retained copies byte-for-byte.
Use the manifest's immutable image and exact source hashes for runtime identity;
the reviewed recipe commit supplies license provenance, not a fresh-provisioning
acceptance result.

`src/tensorfold_dream/*.py` and `src/glm_rsi_dream/attention.py` retain Apache-2.0
headers. The remaining original application code uses the root MIT license.
`tests/upstream_sources.json.gz` contains seven immutable native Python source
fixtures: app, decode, engine, forward, multi, multi_prefill, and verify under
`families/glm5_next/cuda/`. Their SHA-256 hashes match the packaged engine pins.
Tests transform them without rewriting installed source. NOTICE describes the
package's changes. The full upstream notice also preserves credits for its
b12x and other Apache-2.0 integrations; those separate source trees are external.

**License exception to the code-contribution policy:** Contributor Covenant 2.1
is CC BY 4.0 documentation. Its historical license text and attribution are
retained. No other non-permissive borrowed application source is included.
Downloaded weights and runtime binaries are separate prerequisites under
[MODEL-TERMS.md](MODEL-TERMS.md).

## External runtime and Python dependencies

The manifest pins an external NVIDIA-based TensorFold container, a TR3 model
checkpoint, and a DFlash2 drafter. This repository contains no model weights or
container binaries. The download references have not been pulled or loaded in
this publication review. Fresh setup, exact source identity, and native behavior
remain to be qualified on the pair.

Package installation needs Python 3.11 or newer, setuptools, and tokenizers.
The `dev` extra adds pytest, CPU-capable PyTorch, ruff, and build. Native containers
supply CUDA and TensorFold. `tools/prepare_baseline.py` uses Docker and the
container's public huggingface_hub download API. Dependency packages retain their
own licenses; this source tree does not vendor them.

## Fresh pair recipe

Inspect the default offline plan:

```sh
python tools/prepare_baseline.py --config config.example.json
```

Fill config.json with your pair's SSH, fabric, container, API, and protected
service settings. Review the external terms before executing on the head:

```sh
python tools/prepare_baseline.py --config config.json --live
```

Existing container names stop provisioning. The script downloads pinned external
artifacts, creates TP=2 containers, starts rank 1 then rank 0, and monitors memory
and health. The worker command has no API listener options. Direct dashboard,
protected-service, worker-listener, real-reply, and rollback checks remain part
of native qualification. The learning loop separately checks source and device
identity before changing the originals.
