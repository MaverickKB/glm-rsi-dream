# Use the exact stock image. A future source version requires separate qualification.
ARG BASE_IMAGE=ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:bc34d7d63f978cf601f42863b284bc95a567c50c10e9adb0866a635be568bf5f
FROM ${BASE_IMAGE}
COPY dist/glm_dream_loader-0.2.0-py3-none-any.whl /tmp/glm_dream_loader-0.2.0-py3-none-any.whl
RUN python3 -m pip install --no-deps /tmp/glm_dream_loader-0.2.0-py3-none-any.whl
COPY addition/ /opt/dream/addition/
ENV TF_GLM_DREAM_MANIFEST=/opt/dream/addition/manifest.json
# The recipe already calls tensorfold on both ranks. Keep that command surface.
RUN executable="$(command -v tensorfold)" && \
    printf '#!/bin/sh\nexec python3 -m dream_loader.serve "$@"\n' > "$executable" && \
    chmod +x "$executable"
