"""Motor numérico auditable: TRF, SGD y PINN; horizonte adaptable.
Ver simulacion.py para el protocolo y la ejecución sencilla.
"""

from pathlib import Path
import argparse
import copy
import json
import math
import platform
import time

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares
from numba import njit
import torch
from torch import nn

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
OUT = Path(__file__).parent
C = dict(
    rho=0.2,
    S0=0.5,
    mu=0.1,
    sigma_h=0.12,
    sigma_v=0.15,
    h=0.25,
    M=32,
    bank_seed=11272026,
    batch=13,
    sgd_steps=4000,
    sgd_lr=0.0003,
    sgd_decay=500,
    sgd_power=0.6,
    clip=10.0,
    checks_every=20,
    grad_tol=1e-5,
    max_backtracks=25,
    adam_steps=2000,
    adam_lr=0.003,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    trans_batch=32,
    lbfgs_max_iter=2000,
    lbfgs_max_eval=2500,
    lbfgs_history=50,
    lbfgs_grad_tol=1e-7,
    lbfgs_change_tol=1e-10,
    data_weight=100.0,
    ede_weight=0.01,
    recovery_weight=100.0,
    state_min=1e-12,
    variance_min=1e-30,
    network_width=32,
    network_layers=2,
    pinn_dt=1.0,
    trf_max_nfev=400,
    trf_ftol=1e-9,
    trf_xtol=1e-9,
    trf_gtol=1e-7,
)
INITIAL_RATES = [[0.15, 0.20, 1 / 7], [0.25, 0.12, 0.20], [0.10, 0.35, 0.10]]
SEEDS = [90101, 90102, 90103]
LOW, HIGH = np.log([0.02, 0.02, 0.025]), np.log([1.0, 1.0, 0.5])


def metrics(y, p):
    residual = p - y
    denom = np.sum((y - np.mean(y)) ** 2)
    return dict(
        RMSE=float(np.sqrt(np.mean(residual**2))),
        MAE=float(np.mean(abs(residual))),
        R2=float(1 - np.sum(residual**2) / denom) if denom > 0 else np.nan,
        total_observado=float(sum(y)),
        total_predicho=float(sum(p)),
    )


@njit(cache=True)
def simulate(u, x0, noise, scale, h, mu, sigma_h, sigma_v):
    """EM continuo con sensibilidades en logtasas y sin recortes de estados."""
    bh, bv, gamma = np.exp(u)
    steps, M, _ = noise.shape
    steps_week = int(round(7 / h))
    K = steps // steps_week
    pred = np.zeros(K)
    jac = np.zeros((K, 3))
    states = np.zeros((K + 1, 5))
    for m in range(M):
        S, I, V = x0[0], x0[1], x0[4]
        D = np.zeros((3, 3))
        E = np.zeros((3, 3))
        for a in range(5):
            states[0, a] += x0[a] / M
        for k in range(steps):
            w1, w2 = noise[k, m, 0], noise[k, m, 1]
            week = k // steps_week
            q = S * V
            z = (1 - V) * I
            infection = bh * q * h
            dh = infection + sigma_h * q * w1
            recovery = gamma * I * h
            dv = bv * z * h - mu * V * h + sigma_v * z * w2
            pred[week] += scale[week] * infection / M
            for j in range(3):
                dq = D[0, j] * V + S * D[2, j]
                dz = (1 - V) * D[1, j] - I * D[2, j]
                di = bh * dq * h + (infection if j == 0 else 0.0)
                dr = gamma * D[1, j] * h + (recovery if j == 2 else 0.0)
                change = di + sigma_h * dq * w1
                E[0, j] = D[0, j] - change
                E[1, j] = D[1, j] + change - dr
                E[2, j] = (
                    D[2, j]
                    + bv * dz * h
                    + (bv * z * h if j == 1 else 0.0)
                    - mu * D[2, j] * h
                    + sigma_v * dz * w2
                )
                jac[week, j] += scale[week] * di / M
            D, E = E, D
            S -= dh
            I += dh - recovery
            V += dv
            if (
                S < 0
                or I < 0
                or 1 - S - I < -1e-12
                or V < 0
                or V > 1
                or not np.isfinite(S + I + V)
            ):
                return pred, jac, states, False, k, m
            if (k + 1) % steps_week == 0:
                states[week + 1, 0] += S / M
                states[week + 1, 1] += I / M
                states[week + 1, 2] += (1 - S - I) / M
                states[week + 1, 3] += (1 - V) / M
                states[week + 1, 4] += V / M
    return pred, jac, states, True, -1, -1


class Simulator:
    def __init__(self, data, x0, config=C):
        self.c = config
        self.x0 = x0
        self.scale = config["rho"] * data.Nh.to_numpy(float)
        self.noise = np.random.default_rng(config["bank_seed"]).normal(
            0,
            np.sqrt(config["h"]),
            size=(int(len(data) * 7 / config["h"]), config["M"], 2),
        )
        self.calls = 0
        self.rejects = 0
        self.key = None
        self.value = None

    def __call__(self, u):
        if self.key is None or not np.array_equal(self.key, u):
            c = self.c
            self.value = simulate(
                u,
                self.x0,
                self.noise,
                self.scale,
                c["h"],
                c["mu"],
                c["sigma_h"],
                c["sigma_v"],
            )
            self.key = np.array(u).copy()
            self.calls += 1
        if not self.value[3]:
            self.rejects += 1
            raise FloatingPointError(
                f"EM fuera del dominio: paso {self.value[4]}, trayectoria {self.value[5]}"
            )
        return self.value[:3]


def fit_direct(method, data, x0, init, seed):
    """Los dos ajustes directos usan exactamente la misma pérdida y banco."""
    sim = Simulator(data, x0)
    y = data.casos.to_numpy(float)
    K = len(y)
    scale = max(y)
    u = np.log(init)
    history = []
    clips = projections = backtracks = updates = 0
    reason = ""
    success = False
    start = time.perf_counter()
    try:
        if method == "TRF":

            def fun(z):
                try:
                    return (sim(z)[0] - y) / scale * np.sqrt(100 / K)
                except FloatingPointError:
                    return np.full(K, 1e12)

            def jac(z):
                try:
                    return sim(z)[1] / scale * np.sqrt(100 / K)
                except FloatingPointError:
                    return np.zeros((K, 3))

            result = least_squares(
                fun,
                u,
                jac=jac,
                bounds=(LOW, HIGH),
                method="trf",
                max_nfev=C["trf_max_nfev"],
                ftol=C["trf_ftol"],
                xtol=C["trf_xtol"],
                gtol=C["trf_gtol"],
            )
            u = result.x
            updates = result.nfev
            reason = result.message
            success = bool(result.success)
        else:
            rng = np.random.default_rng(seed)
            order = np.array([], dtype=int)
            position = 0
            small = 0
            for step in range(C["sgd_steps"]):
                if position >= K:
                    order = np.array([], dtype=int)
                if not len(order):
                    order = rng.permutation(K)
                    position = 0
                ix = order[position : position + C["batch"]]
                position += C["batch"]
                pred, J, _ = sim(u)
                grad = 100 * (J[ix] / scale).T @ ((pred[ix] - y[ix]) / scale) / len(ix)
                norm = np.linalg.norm(grad)
                if norm > C["clip"]:
                    grad *= C["clip"] / norm
                    clips += 1
                lr = C["sgd_lr"] / (1 + step / C["sgd_decay"]) ** C["sgd_power"]
                accepted = False
                for attempt in range(C["max_backtracks"] + 1):
                    raw = u - lr * grad
                    proposal = np.clip(raw, LOW, HIGH)
                    try:
                        sim(proposal)
                        accepted = True
                        break
                    except FloatingPointError:
                        lr /= 2
                        backtracks += 1
                if not accepted:
                    raise FloatingPointError(
                        "SGD sin propuesta factible tras 25 reducciones"
                    )
                projections += int(np.any(raw != proposal))
                u = proposal
                updates = step + 1
                if updates % C["checks_every"] == 0:
                    pred, J, _ = sim(u)
                    r = (pred - y) / scale
                    gnorm = np.linalg.norm(100 * (J / scale).T @ r / K)
                    history.append(
                        dict(
                            paso=updates,
                            loss=float(50 * np.mean(r * r)),
                            grad_norm=float(gnorm),
                            lr=lr,
                        )
                    )
                    small = small + 1 if gnorm < C["grad_tol"] else 0
                    if small >= 3:
                        success = True
                        reason = "Gradiente completo bajo tolerancia, tres revisiones"
                        break
            if not reason:
                reason = "Presupuesto SGD"
        pred, J, X = sim(u)
        grad = 100 * (J / scale).T @ ((pred - y) / scale) / K
        info = dict(
            estado="completado",
            terminacion=reason,
            convergencia=success,
            segundos=time.perf_counter() - start,
            actualizaciones=updates,
            evaluaciones=sim.calls,
            rechazos_dominio=sim.rejects,
            recortes_gradiente=clips,
            proyecciones=projections,
            reducciones_paso=backtracks,
            grad_norm=float(np.linalg.norm(grad)),
            beta_h=float(np.exp(u[0])),
            beta_v=float(np.exp(u[1])),
            gamma=float(np.exp(u[2])),
            **metrics(y, pred),
        )
        return info, pred, X, history
    except Exception as exc:
        return (
            dict(
                estado="fallo",
                terminacion=str(exc),
                segundos=time.perf_counter() - start,
                actualizaciones=updates,
                evaluaciones=sim.calls,
                rechazos_dominio=sim.rejects,
                recortes_gradiente=clips,
                reducciones_paso=backtracks,
            ),
            None,
            None,
            history,
        )


class ContinuousPINN(nn.Module):
    def __init__(self, x0, init, seed, horizon):
        super().__init__()
        torch.manual_seed(seed)
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 5)
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("logx0", torch.log(torch.tensor(x0)))
        self.logrates = nn.Parameter(torch.log(torch.tensor(init)))

    def forward(self, t):
        # Una sola función de tiempo; ninguna rama depende del año.
        logits = self.logx0 + t / (7 + t) * self.net(2 * t / self.horizon - 1)
        return torch.cat(
            [torch.softmax(logits[:, :3], dim=1), torch.softmax(logits[:, 3:], dim=1)],
            dim=1,
        )


def fit_pinn(data, x0, init, seed, scenario, train_indices, physics_weeks):
    model = ContinuousPINN(x0, init, seed, len(data) * 7)
    y = data.casos.to_numpy(float)
    scale = float(
        max(y[train_indices])
    )  # Nunca se usa el máximo del conjunto reservado.
    observed = torch.tensor(y)
    population = torch.tensor(data.Nh.to_numpy(float))
    time_grid = torch.arange(len(data) * 7 + 1.0).reshape(-1, 1)
    n_transitions = physics_weeks * 7
    rng = np.random.default_rng(seed)
    calls = adam_steps = lb_iterations = lb_calls = 0
    history = []
    best = None
    phase = "inicio"
    status = "completado"
    reason = ""
    start = time.perf_counter()

    def objective(weeks=None, transitions=None):
        nonlocal calls, best
        calls += 1
        X = model(time_grid)
        bh, bv, gamma = model.logrates.exp()
        S, I, R, SV, V = X.T
        # Controles solamente en el dominio de entrenamiento. Evaluar la red futura
        # no introduce observaciones ni pérdidas en las 13 semanas reservadas.
        active = X[: n_transitions + 1]
        if not torch.isfinite(active).all() or active.min().item() < C["state_min"]:
            raise FloatingPointError("PINN fuera del dominio de entrenamiento")
        flux = bh * S * V
        prediction = (
            ((flux[:-1] + flux[1:]) / 2).reshape(len(data), 7).sum(dim=1)
            * C["rho"]
            * population
        )
        take = slice(0, n_transitions)
        eh = (I[1:] - I[:-1] - (bh * S[:-1] * V[:-1] - gamma * I[:-1]))[take]
        ev = (V[1:] - V[:-1] - (bv * SV[:-1] * I[:-1] - C["mu"] * V[:-1]))[take]
        vh = (C["sigma_h"] ** 2 * (S[:-1] * V[:-1]) ** 2)[take]
        vv = (C["sigma_v"] ** 2 * (SV[:-1] * I[:-1]) ** 2)[take]
        if min(vh.min().item(), vv.min().item()) < C["variance_min"]:
            raise FloatingPointError(
                "Varianza PINN inferior al umbral; no se añade piso"
            )
        wx = train_indices if weeks is None else weeks
        tx = slice(None) if transitions is None else transitions
        ld = 0.5 * torch.mean(((prediction[wx] - observed[wx]) / scale) ** 2)
        ede = 0.5 * torch.mean((eh**2 / vh + vh.log() + ev**2 / vv + vv.log())[tx])
        rec = 0.5 * torch.mean(((R[1:] - R[:-1] - gamma * I[:-1])[take][tx]) ** 2)
        total = (
            C["data_weight"] * ld + C["ede_weight"] * ede + C["recovery_weight"] * rec
        )
        if not torch.isfinite(total):
            raise FloatingPointError("Pérdida no finita")
        if weeks is None and (best is None or total.item() < best[0]):
            best = (total.item(), copy.deepcopy(model.state_dict()))
        return (
            total,
            prediction,
            X,
            dict(Ld=ld.item(), LEDE=ede.item(), Lrec=rec.item()),
        )

    try:
        objective()  # checkpoint inicial factible y contabilidad explícita
        adam = torch.optim.Adam(
            model.parameters(),
            lr=C["adam_lr"],
            betas=(C["adam_beta1"], C["adam_beta2"]),
            eps=C["adam_eps"],
            weight_decay=0,
        )
        phase = "Adam"
        for step in range(C["adam_steps"]):
            ix = rng.choice(
                train_indices, min(C["batch"], len(train_indices)), replace=False
            )
            tx = rng.choice(n_transitions, C["trans_batch"], replace=False)
            adam.zero_grad()
            loss, *_ = objective(ix, tx)
            loss.backward()
            adam.step()
            adam_steps = step + 1
            if adam_steps % 100 == 0:
                full, _, _, parts = objective()
                history.append(
                    dict(fase="Adam", paso=adam_steps, loss=full.item(), **parts)
                )
                if adam_steps % 500 == 0:
                    print(
                        scenario,
                        seed,
                        "Adam",
                        adam_steps,
                        round(full.item(), 6),
                        flush=True,
                    )
        phase = "L-BFGS"
        opt = torch.optim.LBFGS(
            model.parameters(),
            lr=1,
            max_iter=C["lbfgs_max_iter"],
            max_eval=C["lbfgs_max_eval"],
            history_size=C["lbfgs_history"],
            tolerance_grad=C["lbfgs_grad_tol"],
            tolerance_change=C["lbfgs_change_tol"],
            line_search_fn="strong_wolfe",
        )

        def closure():
            nonlocal lb_calls
            lb_calls += 1
            opt.zero_grad()
            loss, _, _, parts = objective()
            loss.backward()
            if lb_calls % 100 == 0:
                history.append(
                    dict(
                        fase="L-BFGS evaluaciones",
                        paso=adam_steps + lb_calls,
                        loss=loss.item(),
                        **parts,
                    )
                )
            return loss

        opt.step(closure)
        state = opt.state[next(iter(model.parameters()))]
        lb_iterations = state.get("n_iter", 0)
        reason = "Parada L-BFGS: revisar gradiente y presupuesto"
    except Exception as exc:
        status = "interrumpido"
        reason = f"{phase}: {exc}; se restaura el mejor punto factible de entrenamiento"
        if best is not None:
            model.load_state_dict(best[1])
        if "opt" in locals():
            lb_iterations = opt.state[next(iter(model.parameters()))].get("n_iter", 0)
    try:
        model.zero_grad()
        loss, pred, X, parts = objective()
        loss.backward()
        gradient = torch.cat([a.grad.flatten() for a in model.parameters()])
        gmax = gradient.abs().max().item()
        values = pred.detach().numpy()
        states = X.detach().numpy()
        rates = model.logrates.exp().detach().numpy()
        converged = status == "completado" and gmax <= C["lbfgs_grad_tol"]
        if converged:
            reason = "Tolerancia de gradiente L-BFGS"
        elif status == "completado" and (
            lb_iterations >= C["lbfgs_max_iter"] or lb_calls >= C["lbfgs_max_eval"]
        ):
            reason = "Presupuesto L-BFGS"
        row = dict(
            estado=status,
            terminacion=reason,
            convergencia=converged,
            segundos=time.perf_counter() - start,
            beta_h=rates[0],
            beta_v=rates[1],
            gamma=rates[2],
            loss=loss.item(),
            grad_max=gmax,
            adam_pasos=adam_steps,
            lbfgs_iteraciones=lb_iterations,
            lbfgs_evaluaciones=lb_calls,
            evaluaciones=calls,
            datos_entrenamiento=len(train_indices),
            transiciones=n_transitions,
            escala_datos=scale,
            lote_datos=C["batch"],
            lote_transiciones=C["trans_batch"],
            epocas_equivalentes=adam_steps * C["batch"] / len(train_indices),
            **parts,
        )
        row.update(
            {
                "entrenamiento_" + k: v
                for k, v in metrics(y[train_indices], values[train_indices]).items()
            }
        )
        if physics_weeks == 143:
            row.update(
                {"reserva_" + k: v for k, v in metrics(y[143:], values[143:]).items()}
            )
        row.update(metrics(y, values))
        if not np.isfinite(values).all():
            row["estado"] = "fallo"
            row["terminacion"] = "Predicción no finita"
        torch.save(model.state_dict(), OUT / f"PINN_{scenario}_{seed}.pt")
        return row, values, states, history
    except Exception as exc:
        return (
            dict(
                estado="fallo",
                terminacion=str(exc),
                segundos=time.perf_counter() - start,
                adam_pasos=adam_steps,
                lbfgs_iteraciones=lb_iterations,
                lbfgs_evaluaciones=lb_calls,
                evaluaciones=calls,
            ),
            None,
            None,
            history,
        )
