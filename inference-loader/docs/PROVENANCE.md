# Source identity and provenance

Original loader implementation: this lane, MIT, using the existing shared
expert arithmetic and stock source insertion contract. The separate wheel
carries no upstream engine source and no DREAM training code.

The packaged `pins.json` is derived from LOOP2 commit `6d7c873` resource
`src/glm_rsi_dream/resources/manifest.json`, including its immutable Mia image
and exact 22-file source profile. The owning checkout's `UPSTREAM.md` and `NOTICE` describe the original upstream
source fixtures. Required TensorFold/Mia license and notice files are retained
in this standalone distribution under `licenses/`. The parent files remain
unchanged.

`tests/dream_native_reference.txt` is an exact test-only copy of
`src/glm_rsi_dream/neurons/native.py` from local dream-neurons commit
`e198c2d83d05759734afeb0793e9257873f57045`. Its SHA-256 is asserted by the parity
test. It contains native inference/capture reference code and is excluded from
the wheel. No reference fitting code is copied. `tests/upstream_sources.json.gz`
in this distribution retains the two immutable stock source bodies for forward
and engine, each checked against the loader pins before transformation. The
source distribution includes the parity fixtures; the inference wheel excludes
them. Standalone source tests do not depend on the DREAM engine package.

`tests/dream_four_stream_reference.txt` is the owner's native source captured
during the 02:02 CPU addendum at HEAD
`c6ba875396b5ccb75e9addae90c349ce094d8029`. SHA-256:
`c704f3f7e4d64be5db3a02d0b6e37f102d0f6db0f6aee813d311e2ccd428d2ea`.
The four-stream parity probe asserts that exact source identity. The reference
is test-only and excluded from the wheel. Its geometry, stream orientation,
separate rank sums and BF16 branch boundary match sections 2 and 3 of
`ARCH-deepdive/SPEC-4stream.md`, published for the same 02:02 addendum.
The paired correction exchange uses contiguous flattened FP32 buffers, like
the pinned stock FFN exchange. CPU transport doubles enforce that shape.
Native promotion still requires owner base/study identity, clean sequence
evidence, CUDA numerical validation and the separate deployment grant.

The reviewed public recipe commit supplies hosting integration evidence. The
packaged full source hashes determine executable compatibility. A recipe
repository commit alone does not certify an arbitrary image's installed files.
