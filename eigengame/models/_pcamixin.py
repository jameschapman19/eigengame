import jax.numpy as jnp
import numpy as np
from eigengame.metrics import (_correct_eigenvector_streak,
                               _normalized_subspace_distance)
from jax import jit
from eigengame.metrics import (
    _correct_eigenvector_streak,
    _sum_cosine_similarities,
)

class _PCAMixin:
    def _init_ground_truth(self):
        correct_V, _, _ = np.linalg.svd(self.X.T @ self.X)
        self.correct_V = correct_V[:, : self.config.n_components]
        self.TV_train = _TV(self.correct_V.T, self.X)
        self.TV_val = _TV(self.correct_V.T, self.X_val)

    def _get_scalars(self, global_step):
        scalars = {}
        scalars["examples"] = (global_step[0] + 1) * self.config.batch_size
        scalars["TV train"] = _TV(self._V, self.X)
        scalars["PV train"] = scalars["TV train"] / self.TV_train
        scalars["TV val"] = _TV(self._V, self.X_val)
        scalars["PV val"] = scalars["TV val"] / self.TV_val
        scalars["correct y"] = _correct_eigenvector_streak(self._V, self.correct_V)
        scalars["sum cosine similarities y"] = _sum_cosine_similarities(
            self._V, self.correct_V
        )
        return scalars


@jit
def _TV(U, X_val):
    dof = X_val.shape[0]
    Zx = X_val @ U.T
    return jnp.sum(jnp.linalg.svd(Zx.T @ Zx)[1]) / dof
