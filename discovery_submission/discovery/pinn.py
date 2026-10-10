from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .equations import Equation, term_values


DTYPE = torch.float64


def tensor(value, gradient=False):
    return torch.as_tensor(np.asarray(value), dtype=DTYPE).reshape(-1, 1).clone().requires_grad_(gradient)


class Field(nn.Module):
    def __init__(self, modes=4, width=32, time_end=1.0, length=1.0):
        super().__init__()
        self.modes, self.width, self.time_end, self.length = modes, width, time_end, length
        self.net = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, 2 * modes + 1)).to(dtype=DTYPE)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def basis(self, x, order=0):
        columns = [torch.ones_like(x) if order == 0 else torch.zeros_like(x)]
        for k in range(1, self.modes + 1):
            frequency = 2 * np.pi * k / self.length
            angle = frequency * x + order * np.pi / 2
            columns.extend([frequency**order * torch.cos(angle), frequency**order * torch.sin(angle)])
        return torch.cat(columns, dim=1)

    def forward(self, x, t):
        amplitudes = self.net(2 * t / self.time_end - 1)
        return (amplitudes * self.basis(x)).sum(dim=1, keepdim=True)

    def derivatives(self, x, t, create_graph=True):
        amplitudes = self.net(2 * t / self.time_end - 1)
        u = (amplitudes * self.basis(x)).sum(dim=1, keepdim=True)
        ut = torch.autograd.grad(u.sum(), t, create_graph=create_graph, retain_graph=True)[0]
        ux = (amplitudes * self.basis(x, 1)).sum(dim=1, keepdim=True)
        uxx = (amplitudes * self.basis(x, 2)).sum(dim=1, keepdim=True)
        return u, ut, ux, uxx

    def predict(self, x, t):
        x, t = np.broadcast_arrays(x, t)
        with torch.no_grad():
            output = self(tensor(x), tensor(t)).numpy()
        return output.reshape(x.shape)


class Coefficients(nn.Module):
    def __init__(self, equation, trainable=True):
        super().__init__()
        self.terms = equation.terms
        self.trainable = trainable
        raw = list(equation.coefficients)
        if trainable and "u_xx" in self.terms:
            index = self.terms.index("u_xx")
            value = max(raw[index], 1e-5)
            raw[index] = float(value + np.log(-np.expm1(-value)))
        self.raw = nn.Parameter(torch.tensor(raw, dtype=DTYPE), requires_grad=trainable)

    def values(self):
        return [F.softplus(self.raw[i]) if self.trainable and t == "u_xx" else self.raw[i] for i, t in enumerate(self.terms)]

    def equation(self):
        return Equation(self.terms, tuple(float(v.detach()) for v in self.values()))


def train(data, equation=None, initial=None, seed=11, steps=1200, lbfgs=300, modes=4, width=32,
          bootstrap=False, initial_state=None, adaptive=True):
    if steps < 1 or lbfgs < 0 or modes < 0 or width < 1:
        raise ValueError("Training budgets and architecture sizes must be valid.")
    if equation is not None and not equation.admissibility()[0]:
        raise ValueError("The candidate equation is not admissible for forward training.")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    end, length = float(data["time_end"]), float(data["length"])
    field = Field(modes, width, end, length)
    if initial_state is not None:
        field.load_state_dict(initial_state)
    coefficients = Coefficients(equation, trainable=initial is None) if equation is not None else None
    index = np.arange(len(data.get("u_train", [])))
    if bootstrap and len(index):
        index = rng.choice(index, len(index), replace=True)
    has_data = initial is None and len(index) > 0
    xd = tensor(data["x_train"][index]) if has_data else None
    td = tensor(data["t_train"][index]) if has_data else None
    yd = tensor(data["u_train"][index]) if has_data else None
    scale = max(float(np.sqrt(np.mean(data["u_train"]**2))) if has_data else 0.5, 0.1)
    xc = tensor(rng.uniform(0, length, 700))
    tc = tensor(rng.uniform(0, end, 700), gradient=True)
    xi = tensor(np.linspace(0, length, 64, endpoint=False))
    ti = torch.zeros_like(xi)
    yi = tensor(initial(xi.numpy().ravel())) if initial is not None else None
    parameters = list(field.parameters()) + ([] if coefficients is None else list(coefficients.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=0.003)
    history = []
    resampling = []
    started = time.perf_counter()

    def losses(physics_weight=1.0):
        data_loss = ((field(xd, td) - yd)**2).mean() / scale**2 if has_data else torch.zeros((), dtype=DTYPE)
        ic_loss = ((field(xi, ti) - yi)**2).mean() / scale**2 if yi is not None else torch.zeros((), dtype=DTYPE)
        pde_loss = torch.zeros((), dtype=DTYPE)
        if coefficients is not None:
            u, ut, ux, uxx = field.derivatives(xc, tc)
            rhs = sum(c * v for c, v in zip(coefficients.values(), term_values(u, ux, uxx, coefficients.terms)))
            pde_loss = ((ut - rhs) * end / scale).square().mean()
        return 50 * data_loss + 50 * ic_loss + physics_weight * pde_loss, data_loss, ic_loss, pde_loss

    for step in range(steps):
        if adaptive and coefficients is not None and step in (steps // 2, 3 * steps // 4):
            px, pt = rng.uniform(0, length, 2100), rng.uniform(0, end, 2100)
            u, ut, ux, uxx = field.derivatives(tensor(px), tensor(pt, gradient=True), create_graph=False)
            rhs = sum(c * v for c, v in zip(coefficients.values(), term_values(u, ux, uxx, coefficients.terms)))
            residual = (ut - rhs).detach().numpy().ravel()**2
            if not np.isfinite(residual).all():
                raise RuntimeError("Nonfinite residual during adaptive collocation.")
            probabilities = 0.5 / len(residual) + 0.5 * (residual + 1e-20) / (residual.sum() + len(residual) * 1e-20)
            chosen = rng.choice(len(px), 700, replace=False, p=probabilities)
            xc, tc = tensor(px[chosen]), tensor(pt[chosen], gradient=True)
            resampling.append({"step": step, "pool_residual_mse": float(residual.mean()),
                               "selected_residual_mse": float(residual[chosen].mean()), "points": len(chosen)})
        optimizer.zero_grad(set_to_none=True)
        weight = 0.0 if has_data and step < steps // 4 else 1.0
        total, dl, il, pl = losses(weight)
        if not torch.isfinite(total):
            raise RuntimeError("Nonfinite PINN training loss.")
        total.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            history.append({"step": step, "data": float(dl.detach()), "initial": float(il.detach()),
                            "pde": float(pl.detach()), "total": float(total.detach())})
    if lbfgs:
        optimizer = torch.optim.LBFGS(parameters, max_iter=lbfgs, history_size=50,
                                      tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad(set_to_none=True)
            total = losses()[0]
            if not torch.isfinite(total):
                raise RuntimeError("Nonfinite PINN refinement loss.")
            total.backward()
            return total

        optimizer.step(closure)
    total, dl, il, pl = losses()
    if not torch.isfinite(total):
        raise RuntimeError("Nonfinite final training objective.")
    summary = {"seed": seed, "seconds": time.perf_counter() - started,
               "train_data_loss": float(dl.detach()) * scale**2,
               "train_initial_loss": float(il.detach()) * scale**2,
               "train_pde_loss": float(pl.detach()) * (scale / end)**2,
               "final_objective": float(total.detach()), "history": history, "adaptive_collocation": resampling,
               "collocation_rule": "half uniform, half residual-energy sampling" if adaptive else "fixed uniform sampling",
               "mode": "forward" if initial is not None else ("inverse" if equation is not None else "neural"),
               "physics_end": end, "parameter_count": sum(p.numel() for p in parameters)}
    return field, None if coefficients is None else coefficients.equation(), summary


def save_model(path, field, equation=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": field.state_dict(), "modes": field.modes, "width": field.width,
                "time_end": field.time_end, "length": field.length,
                "equation": None if equation is None else equation.as_dict()}, path)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    field = Field(checkpoint["modes"], checkpoint["width"], checkpoint["time_end"], checkpoint["length"])
    field.load_state_dict(checkpoint["state"])
    field.eval()
    return field, None if checkpoint["equation"] is None else Equation.from_dict(checkpoint["equation"])
