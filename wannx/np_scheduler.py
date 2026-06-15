"""Pure-numpy FlowMatch UniPC multistep scheduler (no torch).

Faithful port of the WAN `FlowUniPCMultistepScheduler` for the inference defaults:
solver_order=2, predict_x0=True, solver_type='bh2', prediction_type='flow_prediction',
lower_order_final=True, final_sigmas_type='zero', use_dynamic_shifting=False. Lets the
ONNX-Runtime inference path run without importing torch.
"""
import numpy as np


class NumpyUniPCScheduler:
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0,
                 solver_order: int = 2):
        self.num_train_timesteps = num_train_timesteps
        self.shift = float(shift)
        self.solver_order = solver_order
        self.predict_x0 = True

    # WAN flow-matching: alpha_t = 1 - sigma, sigma_t = sigma
    @staticmethod
    def _alpha_sigma(sigma):
        return 1.0 - sigma, sigma

    def set_timesteps(self, num_steps: int, shift: float = None):
        if shift is not None:
            self.shift = float(shift)
        # training sigma range (alphas = linspace(1, 1/N, N)[::-1]; sigmas = 1 - alphas)
        alphas = np.linspace(1, 1.0 / self.num_train_timesteps, self.num_train_timesteps)[::-1]
        train_sigmas = 1.0 - alphas
        sigma_max, sigma_min = float(train_sigmas[0]), float(train_sigmas[-1])
        sigmas = np.linspace(sigma_max, sigma_min, num_steps + 1)[:-1].astype(np.float64)
        s = self.shift
        sigmas = s * sigmas / (1 + (s - 1) * sigmas)            # flow-shift
        # torch UniPC casts timesteps to int64; truncate identically so the DiT
        # embedding receives the same values, then back to float32 for ORT.
        self.timesteps = (sigmas * self.num_train_timesteps).astype(np.int64).astype(np.float32)
        self.sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float64)  # final_sigmas='zero'
        self.num_inference_steps = num_steps
        self._step_index = 0
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = 1
        return self.timesteps

    def scale_noise(self, latents):       # WAN starts from raw N(0,1) noise
        return latents

    def _convert(self, model_output, sample, si):
        # predict_x0, flow_prediction: x0 = sample - sigma_t * model_output
        return sample - self.sigmas[si] * model_output

    def _bh_coeffs(self, order, h, rks):
        hh = -h                          # predict_x0
        h_phi_1 = np.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        B_h = np.expm1(hh)               # bh2
        R, b = [], []
        factorial_i = 1
        for i in range(1, order + 1):
            R.append(np.power(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i
        return np.stack(R), np.array(b, dtype=np.float64), h_phi_1, B_h

    def _uni_p(self, model_output, sample, order):
        si = self._step_index
        m0 = self.model_outputs[-1]
        x = sample.astype(np.float64)
        sigma_t, sigma_s0 = self.sigmas[si + 1], self.sigmas[si]
        alpha_t, sigma_t = self._alpha_sigma(sigma_t)
        alpha_s0, sigma_s0 = self._alpha_sigma(sigma_s0)
        lambda_t = np.log(alpha_t) - np.log(sigma_t)
        lambda_s0 = np.log(alpha_s0) - np.log(sigma_s0)
        h = lambda_t - lambda_s0
        rks, D1s = [], []
        for i in range(1, order):
            mi = self.model_outputs[-(i + 1)]
            ssi = self.sigmas[si - i]
            a_si, s_si = self._alpha_sigma(ssi)
            lambda_si = np.log(a_si) - np.log(s_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi.astype(np.float64) - m0) / rk)
        rks.append(1.0)
        rks = np.array(rks, dtype=np.float64)
        R, b, h_phi_1, B_h = self._bh_coeffs(order, h, rks)
        if D1s:
            D1s = np.stack(D1s, axis=1)              # (B, K, ...)
            rhos_p = np.array([0.5]) if order == 2 else np.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s, rhos_p = None, None
        x_t = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = np.einsum("k,bk...->b...", rhos_p, D1s)
            x_t = x_t - alpha_t * B_h * pred_res
        return x_t

    def _uni_c(self, this_model_output, last_sample, this_sample, order):
        si = self._step_index
        m0 = self.model_outputs[-1]
        x = last_sample.astype(np.float64)
        model_t = this_model_output.astype(np.float64)
        sigma_t, sigma_s0 = self.sigmas[si], self.sigmas[si - 1]
        alpha_t, sigma_t = self._alpha_sigma(sigma_t)
        alpha_s0, sigma_s0 = self._alpha_sigma(sigma_s0)
        lambda_t = np.log(alpha_t) - np.log(sigma_t)
        lambda_s0 = np.log(alpha_s0) - np.log(sigma_s0)
        h = lambda_t - lambda_s0
        rks, D1s = [], []
        for i in range(1, order):
            mi = self.model_outputs[-(i + 1)]
            ssi = self.sigmas[si - (i + 1)]
            a_si, s_si = self._alpha_sigma(ssi)
            lambda_si = np.log(a_si) - np.log(s_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi.astype(np.float64) - m0) / rk)
        rks.append(1.0)
        rks = np.array(rks, dtype=np.float64)
        R, b, h_phi_1, B_h = self._bh_coeffs(order, h, rks)
        if D1s:
            D1s = np.stack(D1s, axis=1)
        else:
            D1s = None
        rhos_c = np.array([0.5]) if order == 1 else np.linalg.solve(R, b)
        x_t = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        corr = np.einsum("k,bk...->b...", rhos_c[:-1], D1s) if D1s is not None else 0
        D1_t = model_t - m0
        x_t = x_t - alpha_t * B_h * (corr + rhos_c[-1] * D1_t)
        return x_t

    def step(self, model_output, sample, step_index=None):
        mo = np.asarray(model_output, dtype=np.float64)
        x = np.asarray(sample, dtype=np.float64)
        si = self._step_index
        use_corrector = si > 0 and self.last_sample is not None
        # final step has sigma_t == 0 -> log(0); the 0/inf terms cancel (matches
        # torch bit-for-bit), so silence the benign warning.
        np.seterr(divide="ignore")
        mo_conv = self._convert(mo, x, si)
        if use_corrector:
            x = self._uni_c(mo_conv, self.last_sample, x, self.this_order)
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = mo_conv
        this_order = min(self.solver_order, len(self.timesteps) - si)  # lower_order_final
        self.this_order = min(this_order, self.lower_order_nums + 1)
        self.last_sample = x
        prev = self._uni_p(mo, x, self.this_order)
        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        self._step_index += 1
        return prev.astype(np.float32)
