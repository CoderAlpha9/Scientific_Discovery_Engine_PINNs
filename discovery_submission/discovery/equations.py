from dataclasses import dataclass
from itertools import combinations

import numpy as np


TERMS = ("u_xx", "u_x", "u", "u2", "u3", "u*u_x", "1")
DIMENSIONS = {
    "u_xx": (1, -2, 0), "u_x": (1, -1, 0), "u": (1, 0, 0),
    "u2": (2, 0, 0), "u3": (3, 0, 0), "u*u_x": (2, -1, 0), "1": (0, 0, 0),
}
TARGET_DIMENSION = (1, 0, -1)


def coefficient_dimension(term):
    return tuple(a - b for a, b in zip(TARGET_DIMENSION, DIMENSIONS[term]))


def term_values(u, ux, uxx, terms):
    values = {"u_xx": uxx, "u_x": ux, "u": u, "u2": u**2,
              "u3": u**3, "u*u_x": u * ux, "1": u * 0 + 1}
    return [values[term] for term in terms]


@dataclass(frozen=True)
class Equation:
    terms: tuple
    coefficients: tuple

    def __post_init__(self):
        if not self.terms or len(self.terms) != len(self.coefficients):
            raise ValueError("Each nonempty equation needs one coefficient per term.")
        if len(set(self.terms)) != len(self.terms) or any(t not in TERMS for t in self.terms):
            raise ValueError("Unknown or repeated equation term.")
        if not np.isfinite(self.coefficients).all():
            raise ValueError("Equation coefficients must be finite.")

    def rhs(self, u, ux, uxx):
        return sum(c * v for c, v in zip(self.coefficients, term_values(u, ux, uxx, self.terms)))

    def admissibility(self, units=None, forward=True):
        if units is not None:
            for term in self.terms:
                if term not in units or tuple(units[term]) != coefficient_dimension(term):
                    return False, "dimensional inconsistency"
        if forward and "u_xx" in self.terms and self.coefficients[self.terms.index("u_xx")] < 0:
            return False, "backward diffusion is not a stable forward model in this regime"
        return True, "admissible"

    def as_dict(self):
        return {"terms": list(self.terms), "coefficients": list(self.coefficients),
                "coefficient_dimensions_U_L_T": {t: list(coefficient_dimension(t)) for t in self.terms},
                "expression": "u_t = " + " + ".join(f"({c:.8g})*{t}" for t, c in zip(self.terms, self.coefficients))}

    @classmethod
    def from_dict(cls, value):
        return cls(tuple(value["terms"]), tuple(float(c) for c in value["coefficients"]))


def supports(terms=TERMS, maximum=4):
    if any(t not in TERMS for t in terms) or not 1 <= maximum <= len(TERMS):
        raise ValueError("Invalid equation grammar.")
    ordered = tuple(t for t in TERMS if t in terms)
    return [s for n in range(1, min(maximum, len(ordered)) + 1) for s in combinations(ordered, n)]


def canonical_support(terms):
    if any(t not in TERMS for t in terms):
        raise ValueError("Term is outside the supported grammar.")
    return tuple(t for t in TERMS if t in set(terms))
