# Third-party artifacts used at image build time

Both are unmodified copies of upstream artifacts of b12x (a PyPI release and a GitHub commit archive) (github.com/local-inference-lab/b12x, Apache License 2.0;
see NOTICE for the other licences inside the snapshot). Verify with `sha256sum -c` against the values below.

| file | what | sha256 |
|---|---|---|
| `b12x-1.3.0-py3-none-any.whl` | the PyPI release `b12x==1.3.0` (https://pypi.org/project/b12x/1.3.0/): the package whose compiler and runtime helpers the RoCE runtime imports | `c97d88635521a7fdd4f67717c1835a017d0dcea49d49cc7aa5b4299f290d19e3` |
| `b12x-b58f34ea.tgz` | GitHub archive of the repository at commit `b58f34eaf978277621efced6678e6713fd7122e4` (https://github.com/local-inference-lab/b12x/archive/b58f34eaf978277621efced6678e6713fd7122e4.tar.gz): the revision of the `comm/roce` one-shot RDMA runtime with `API_VERSION == 1`, which overlay/roce_collectives.py requires | `8cfd2d8bf09169d00a669f72f179c54bdfc49af35b05214a5e5f0fdb1d5779d8` |

The Dockerfile installs the wheel with `pip install --no-deps`, then copies `b12x/comm/roce` and `b12x/comm/__init__.py`
from the snapshot over the installed package, builds the runtime's proxy so no compiler runs at start, and asserts the
API version. Newer b12x revisions change the runtime's API; bump the pin only together with the adapter.
