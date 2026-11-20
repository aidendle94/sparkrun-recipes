# Inkling-2-Small on 4x DGX Spark with SGLang: the hooks image.
# Base: the SGLang team's lmsysorg/sglang:dev-inkling-2-small (Apache-2.0), unmodified and pinned by digest (the tag as
# of 2026-11-12, arm64). The one layer on top holds this recipe's files:
#   /patches                      the hooks (on PYTHONPATH, so the interpreter imports patches/sitecustomize.py at
#                                 start), the confidence head's file included
#   /opt/inkling2-spark           the launcher, the client-side checks, the docs, kernels/ and the licence files
#   /opt/inkling2-spark/prebuilt  spark_csf_gemm_sm120.so, the forked grouped GEMM of the packed block scales, built by
#                                 kernels/csf_gemm/build_csf_gemm.py from this same base image (README, Run it, step 2)
# Nothing is compiled or installed while the image is built, and no hook is switched on here: the launcher sets the
# SPARK_* variables, so every switch of the README still turns off by setting its launcher variable empty. The
# launcher mounts its own copy of patches/ over /patches, so under the launcher an image runs the hooks of the
# clone it is launched from.
# Build on a Spark (arm64), from this directory, after building the module (step 2 of the README):
#   docker build --build-context repo=../../.. --build-context prebuilt="$HOME/inkling2-csf-build" \
#     --build-arg REVISION=$(git rev-parse --short HEAD) -t sglang-inkling2-spark:local .
# Published as aidendle94/sparkrun-sglang-inkling2-gb10:<tag>; the README names the current tag. Every change to the
# recipe gets a new tag.
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
COPY kernels /opt/inkling2-spark/kernels
COPY --from=prebuilt spark_csf_gemm_sm120.so /opt/inkling2-spark/prebuilt/spark_csf_gemm_sm120.so
COPY patches /patches
ENV PYTHONPATH=/patches
