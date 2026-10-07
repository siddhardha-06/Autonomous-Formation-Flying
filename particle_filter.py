"""Particle filter with ANN-informed resampling (paper Section IV-E).

Per time step:
 1. propagate particles through nonlinear dynamics (leader inputs = uniform disturbance)
 2. render synthetic feature locations for every particle; weight = 1 / MSE against the
    detected features (occluded / undetected features are ignored)
 3. estimate = weighted mean
 4. resample: floor(alpha*N) from the weighted particles, N - floor(alpha*N) from
    (ANN pose label + error distribution measured on the validation set)

Extension (known_los): if the gimbal line of sight is supplied, every particle's position is
moved onto that bearing (keeping its range), so the filter only has to resolve range + attitude.
"""
import numpy as np
from config import *
from aircraft import project
from dynamics import propagate


class ParticleFilter:
    def __init__(self, n, error_pool, alpha=0.9, alpha_every=1, likelihood="inv_mse",
                 sigma_px=5.0, seed=0):
        self.n, self.pool = n, error_pool
        self.alpha, self.alpha_every = alpha, alpha_every
        self.likelihood, self.sigma_px = likelihood, sigma_px
        self.rng = np.random.default_rng(seed)
        self.sig_los = np.radians(LOS_NOISE_DEG)
        self.k = 0
        self.x = None

    # particles drawn around the ANN label using the empirical (true - classified) error
    def sample_from_label(self, label_pose, m):
        d = self.pool[self.rng.integers(len(self.pool), size=m)]
        lim = np.radians(ATT_LIMIT_DEG)
        s = np.empty((m, 6))
        s[:, :3] = label_pose[:3] + d[:, :3]
        s[:, 3] = np.radians(label_pose[3] + d[:, 3])
        s[:, 4:] = self.rng.uniform(-lim, lim, (m, 2))     # pitch/yaw are not in the label
        return s

    def snap_to_los(self, los):
        """Put each particle on the measured bearing (+ per-particle LOS noise), keeping range."""
        u = los[None, :] + self.rng.normal(0.0, self.sig_los, (len(self.x), 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        self.x[:, :3] = np.linalg.norm(self.x[:, :3], axis=1, keepdims=True) * u

    def weights(self, uv_meas, det):
        if det.sum() == 0:
            return np.ones(self.n) / self.n
        pred = project(self.x)                              # (N,14,2)
        mse = ((pred[:, det, :] - uv_meas[det][None]) ** 2).mean(axis=(1, 2))
        if self.likelihood == "gauss":
            ll = -mse / (2.0 * self.sigma_px ** 2)
            w = np.exp(ll - ll.max())
        else:                                               # paper: inverse MSE
            w = 1.0 / (mse + 1e-3)
        return w / w.sum()

    def step(self, u_follower, uv_meas, det, label_pose, los=None):
        if self.x is None:                                  # initialisation
            self.x = self.sample_from_label(label_pose, self.n)
        else:
            self.x = propagate(self.x, u_follower, self.rng)
        if los is not None:
            self.snap_to_los(los)
        w = self.weights(uv_meas, det)
        est = (w[:, None] * self.x).sum(0)

        # --- resampling ---
        a = self.alpha if (self.k % self.alpha_every == 0) else 1.0
        n_pf = int(np.floor(a * self.n))
        cs = np.cumsum(w); cs[-1] = 1.0
        pos = (self.rng.random() + np.arange(n_pf)) / max(n_pf, 1)   # systematic resampling
        idx = np.searchsorted(cs, pos)
        new = self.x[np.minimum(idx, self.n - 1)]
        if self.n - n_pf > 0:
            new = np.vstack([new, self.sample_from_label(label_pose, self.n - n_pf)])
        self.x = new
        self.k += 1
        return est
