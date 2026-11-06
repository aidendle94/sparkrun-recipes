# DeepSeek-V4.1-Flash on 4x DGX Spark with SGLang — overlay image.
# Base: the SGLang team's nightly image (Apache-2.0), unmodified. The overlay adds import hooks (Engram tables from
# node-local NVMe inside the CUDA graph, next-chunk prefetch, the b12x MXFP8 kernel, the SM12x candidate indexer and
# prefill fixes, allocator and page-cache releases, one-shot RDMA collectives, served-model aliases; see README).
# Build on a Spark (arm64):
#   cp -r ../../../licenses . && docker build -t sglang-dsv41-spark:local .   # the licence texts live at the repository root
# The published tag aidendle94/sparkrun-sglang-dsv41-gb10:production-1.8 is this file built on the pinned base.
# Pinned by digest: lmsysorg/sglang:nightly-dev-cu13-20261018-b234b5c3 (SGLang main b234b5c32, 2026-10-17; torch 2.14.1,
# CUDA 13.0, Triton 3.8.0, FlashInfer 0.7.2, sgl-deep-gemm 0.2.2, sglang-kernel 0.4.10). Until production-1.7.1 the base
# was dev-dsv41 (sha256:4a5d132a…, SGLang da64c5cb, 2026-09-11).
ARG BASE=lmsysorg/sglang@sha256:bfef46e5a45a815f8c2f88a722980baf47b71a8689b89c6cc738f596ceeaf47c
FROM ${BASE}
COPY LICENSE NOTICE /opt/dsv41-spark/
COPY licenses /opt/dsv41-spark/licenses   # build context: this directory with ../../../licenses copied in (see README)
COPY overlay /opt/dsv41-spark/overlay
COPY tests /opt/dsv41-spark/tests
COPY vendor /opt/dsv41-spark/vendor
# (vendor/ is the b12x release production-1.7.1 installed; kept for provenance, not installed since production-1.8)
# b12x (Apache-2.0) ships inside the base's FlashInfer 0.7.2 (flashinfer/experimental/b12x, imported as `b12x`; the
# standalone package is archived). Its comm/roce one-shot RDMA runtime is used as is; the proxy is compiled here so no
# compiler runs at start. Never pip-install the standalone b12x over it: both provide the same top-level package.
RUN set -eux; python3 -c "import b12x, flashinfer, os; assert 'flashinfer' in b12x.__file__ or os.path.isdir(os.path.join(os.path.dirname(flashinfer.__file__), 'experimental', 'b12x')), b12x.__file__" \
 && mkdir -p /opt/dsv41-spark/roce-cache \
 && B12X_ROCE_CACHE_DIR=/opt/dsv41-spark/roce-cache python3 -c "from b12x.comm.roce import _proxy; print('roce proxy', _proxy.load())" \
 && python3 -c "from b12x.comm import roce; assert roce.API_VERSION == 1, roce.API_VERSION"
ENV B12X_ROCE_CACHE_DIR=/opt/dsv41-spark/roce-cache
RUN set -eux; cd /opt/dsv41-spark/overlay \
 && gcc -O2 -std=gnu11 -shared -fPIC -pthread -o libengram_rows.so engram_rows.c \
 && python3 /opt/dsv41-spark/tests/test_engram_rows.py \
 && python3 -c "import flashinfer; from flashinfer.gemm import gemm_base; assert 'b12x' in (gemm_base.mm_mxfp8.__doc__ or ''), 'FlashInfer mm_mxfp8 in this base has no b12x backend'; print('flashinfer', flashinfer.__version__, 'mm_mxfp8 offers b12x')"
ENV PYTHONPATH=/opt/dsv41-spark/overlay
