# Inkling-2-Small on 4x DGX Spark with SGLang: the hooks image.
# Base: the SGLang team's lmsysorg/sglang:dev-inkling-2-small (Apache-2.0), unmodified and pinned by digest (the tag as
# of 2026-11-12, arm64). The one layer on top holds this recipe's hooks (patches/, put on PYTHONPATH so that the
# interpreter imports patches/sitecustomize.py at start), the client-side checks and the launcher (bench/, launch/, for reference: they run on rank 0, not in the
# container) and the licence files.
# Nothing is compiled or installed, and no hook is switched on here: the launcher sets the SPARK_* variables, so
# every switch of the README still turns off by setting its launcher variable empty.
# Build on a Spark (arm64), from this directory:
#   docker build --build-context repo=../../.. -t sglang-inkling2-spark:local .
# The published tag aidendle94/sparkrun-sglang-inkling2-gb10:experimental-1.0 is this file built on the pinned base.
# launch/launch-inkling2.sh runs either image: it mounts its own copy of patches/ over /patches.
ARG BASE=lmsysorg/sglang@sha256:424c1da97f6d6eef1612006c77b2ca791657a8184c1fbdbc682ebf9485d3b32c
FROM ${BASE}
ARG REVISION=unknown
LABEL org.opencontainers.image.title="Inkling-2-Small on 4x DGX Spark (SGLang + GB10 hooks)" \
      org.opencontainers.image.source="https://github.com/aidendle94/sparkrun-recipes" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      org.opencontainers.image.base.name="lmsysorg/sglang:dev-inkling-2-small"
COPY --from=repo LICENSE /opt/inkling2-spark/LICENSE
COPY --from=repo licenses /opt/inkling2-spark/licenses
COPY NOTICE README.md recipe.yaml /opt/inkling2-spark/
COPY docs /opt/inkling2-spark/docs
COPY bench /opt/inkling2-spark/bench
COPY launch /opt/inkling2-spark/launch
COPY patches /patches
ENV PYTHONPATH=/patches
