"""PH 306 Week 2: Linear Algebra, Part 1.

Complete each function below. Use NumPy arrays for vector and matrix inputs and
outputs unless the function documentation specifies otherwise.
"""

# --- Imports --- #
# Built-in Libraries
from numpy.typing import NDArray
from typing import Any

# Numerical Libraries
import numpy as np
from astropy import units as u

# Local Utilities
from plotutil import colored_line_between_pts


# Type Hints
Array = NDArray[np.float64]


# Constants
EARTH_GRAVITY = 9.8 * u.m / u.s**2


def commutator(first: Array, second: Array) -> Array:
    """Compute the matrix commutator $[A, B] = AB - BA$."""
    return (first @ second) - (second @ first)

def are_perpendicular(first: Array, second: Array, tolerance=1e-10) -> bool:
    """Determine whether two vectors are perpendicular."""

    AB = first @ second

    return abs(AB) < tolerance 


def are_parallel(first: Array, second: Array, tolerance=1e-10) -> bool:
    """Determine whether two vectors are parallel."""

    dot_product = first @ second
    magnitude_product = np.linalg.norm(first) * np.linalg.norm(second)

    return abs(abs(dot_product) - magnitude_product) < tolerance


def are_commutative(first: Array, second: Array, tolerance=1e-10) -> bool:
    """Determine whether two matrices commute."""
    AB = first @ second
    BA = second @ first
    return abs(AB - BA).max() < tolerance

def is_hermitian(matrix: Array, tolerance=1e-10) -> bool:
    """Determine whether a matrix is Hermitian."""
    if matrix.shape[0] != matrix.shape[1]:
        return False  # Not a square matrix, cannot be Hermitian

    return abs(matrix - matrix.conj().T).max() < tolerance


def is_unitary(matrix: Array, tolerance=1e-10) -> bool:
    """Determine whether a matrix is unitary."""
    raise NotImplementedError("Implement is_unitary")


def is_linear_operator(matrix: Array, tolerance=1e-10) -> bool:
    """Determine whether a matrix represents a linear operator."""
    raise NotImplementedError("Implement is_linear_operator")


def projection(vector_a: Array, vector_b: Array) -> Array:
    """Project vector $\\vec{A}$ onto vector $\\vec{B}$."""
    raise NotImplementedError("Implement projection")


def rotate_vector(vector: Array, axis: Array, theta: float) -> Array:
    """Rotate a vector through angle theta about an axis in $\\mathbb{R}^3$."""
    raise NotImplementedError("Implement rotate_vector")


def plane_from_points(first: Array, second: Array, third: Array) -> tuple[Array, float]:
    """Find the plane through three noncollinear points."""
    raise NotImplementedError("Implement plane_from_points")


def distance_point_to_plane(point: Array, normal: Array, offset: float) -> float:
    """Find the minimum distance from a point to a plane."""
    raise NotImplementedError("Implement distance_point_to_plane")


def distance_between_lines(
        first_point: Array,
        first_direction: Array,
        second_point: Array,
        second_direction: Array) -> float:
    """Find the minimum distance between two lines in $\\mathbb{R}^3$."""
    raise NotImplementedError("Implement distance_between_lines")


def solve_cable_tension(N: int, L: float, rho: float, g: float = EARTH_GRAVITY) -> tuple[Array, Array]:
    """Solve for the tension in a hanging cable discretized into N segments."""
    raise NotImplementedError("Implement solve_cable_tension")


def plot_cable_tension(z: Array, T: Array, L: float) -> Any:
    """Plot the tension along a hanging cable, colored by tension magnitude."""
    raise NotImplementedError("Implement plot_cable_tension")
