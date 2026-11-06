"""PQ-FedVC-KAN: post-quantum secure federated variational-circuit KANs for privacy-preserving heart disease screening."""
from .models import build, n_params, VCKAN, KANLayer
from .train import fed_run, central_run, youden_from_counts, metrics
__version__ = "1.0.0"
