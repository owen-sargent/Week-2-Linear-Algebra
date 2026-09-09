"""PH 306 Week 2: Linear Algebra, Part 1.

Complete each function below. Use NumPy arrays for vector and matrix inputs and
outputs unless the function documentation specifies otherwise.
"""

# --- Imports --- #
# Built-in Libraries
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from typing import Any, Callable

# Numerical Libraries
import numpy as np
from astropy import units as u

# Local Utilities
# (from plotutil import colored_line_between_pts) is not used


# Type Hints
Array = NDArray[np.float64]


# Constants
EARTH_GRAVITY = 9.8 * u.m / u.s**2


def commutator(first: Array, second: Array) -> Array:
    """Compute the matrix commutator $[A, B] = AB - BA$.

    Parameters
    ----------
    first : Array
        The first matrix.
    second : Array
        The second matrix.

    Returns
    -------
    Array
        The commutator [A, B].
    """
    return (first @ second) - (second @ first)


def are_perpendicular(first: Array, second: Array, tolerance=1e-10) -> bool:
    """Determine whether two vectors are perpendicular.

    Parameters
    ----------
    first : Array
        The first vector.
    second : Array
        The second vector.
    tolerance : float, optional
        The tolerance for determining perpendicularity. Default is 1e-10.

    Returns
    -------
    bool
        True if the vectors are perpendicular, False otherwise.
    """

    AB = first @ second

    return abs(AB) < tolerance


def are_parallel(first: Array, second: Array, tolerance=1e-10) -> bool:
    """Determine whether two vectors are parallel.

    Parameters
    ----------
    first : Array
        The first vector.
    second : Array
        The second vector.
    tolerance : float, optional
        The tolerance for determining parallelism. Default is 1e-10.

    Returns
    -------
    bool
        True if the vectors are parallel, False otherwise.
    """

    dot_product = first @ second
    magnitude_product = np.linalg.norm(first) * np.linalg.norm(second)

    return abs(abs(dot_product) - magnitude_product) < tolerance


def are_commutative(first: Array, second: Array, tolerance=1e-10) -> bool:
    """Determine whether two matrices commute.

    Parameters
    ----------
    first : Array
        The first matrix.
    second : Array
        The second matrix.
    tolerance : float, optional
        The tolerance for determining commutativity. Default is 1e-10.

    Returns
    -------
    bool
        True if the matrices are commutative, False otherwise.
    """
    AB = first @ second
    BA = second @ first
    return abs(AB - BA).max() < tolerance


def is_hermitian(matrix: Array, tolerance=1e-10) -> bool:
    """Determine whether a matrix is Hermitian.

    Parameters
    ----------
    matrix : Array
        The matrix to check.
    tolerance : float, optional
        The tolerance for determining Hermiticity. Default is 1e-10.

    Returns
    -------
    bool
        True if the matrix is Hermitian, False otherwise.
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False  # Not a square matrix, cannot be Hermitian

    return abs(matrix - matrix.conj().T).max() < tolerance


def is_unitary(matrix: Array, tolerance=1e-10) -> bool:
    """Determine whether a matrix is unitary.

    Parameters
    ----------
    matrix : Array
        The matrix to check.
    tolerance : float, optional
        The tolerance for determining unitarity. Default is 1e-10.

    Returns
    -------
    bool
        True if the matrix is unitary, False otherwise.
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False  # Not a square matrix, cannot be unitary

    A_H = matrix.conj().T
    identity = np.eye(matrix.shape[0])
    return abs(matrix @ A_H - identity).max() < tolerance


def is_linear_operator(matrix: Array, tolerance=1e-10) -> bool:
    """Determine whether a matrix represents a linear operator.

    Parameters
    ----------
    matrix : Array
        The matrix to check.
    tolerance : float, optional
        The tolerance for determining linearity. Default is 1e-10.

    Returns
    -------
    bool
        True if the matrix represents a linear operator, False otherwise.
    """
    u = np.random.rand(matrix.shape[1])
    v = np.random.rand(matrix.shape[1])
    return abs(matrix @ (u + v) - (matrix @ u + matrix @ v)).max() < tolerance


def projection(vector_a: Array, vector_b: Array) -> Array:
    """Project vector $\\vec{A}$ onto vector $\\vec{B}$.

    Parameters
    ----------
    vector_a : Array
        The vector to be projected.
    vector_b : Array
        The vector onto which to project.

    Returns
    -------
    Array
        The projection of vector_a onto vector_b.
    """
    dot_ab = vector_a @ vector_b
    dot_bb = vector_b @ vector_b
    return (dot_ab / dot_bb) * vector_b


def rotate_vector(vector: Array, axis: int, theta: float) -> Array:
    """Rotate a vector through angle theta about an axis in R^3.

    Parameters
    ----------
    vector : Array
        The vector to rotate.
    axis : int
        The axis about which to rotate (0, 1, or 2).
    theta : float
        The angle of rotation in radians.

    Returns
    -------
    Array
        The rotated vector.
    """

    if axis == 0:
        k = np.array([1.0, 0.0, 0.0])
    elif axis == 1:
        k = np.array([0.0, 1.0, 0.0])
    elif axis == 2:
        k = np.array([0.0, 0.0, 1.0])
    else:
        raise ValueError("Axis must be 0, 1, or 2.")

    v_cos = vector * np.cos(theta)
    v_sin = np.cross(k, vector) * np.sin(theta)
    v_dot = k * (k @ vector) * (1 - np.cos(theta))

    return v_cos + v_sin + v_dot


def plane_from_points(first: Array, second: Array, third: Array) -> tuple[Array, float]:
    """Find the plane through three noncollinear points.

    Parameters
    ----------
    first : Array
        The first point.
    second : Array
        The second point.
    third : Array
        The third point.

    Returns
    -------
    tuple[Array, float]
        The normal vector and offset of the plane.
    """
    if np.linalg.matrix_rank(np.vstack([first, second, third])) < 3:
        raise ValueError("The three points must be noncollinear.")
    u = second - first
    v = third - first
    normal_cross = np.cross(u, v)
    normal = normal_cross / np.linalg.norm(normal_cross)
    offset = normal @ first
    return normal, offset


def distance_point_to_plane(point: Array, normal: Array, offset: float) -> float:
    """Find the minimum distance from a point to a plane.

    Parameters
    ----------
    point : Array
        The point from which to find the distance.
    normal : Array
        The normal vector of the plane.
    offset : float
        The offset of the plane.

    Returns
    -------
    float
        The minimum distance from the point to the plane.
    """
    distance = abs(normal @ point + offset) / np.linalg.norm(normal)
    return distance


def distance_between_lines(
        point1: Array,
        direction1: Array,
        point2: Array,
        direction2: Array
        ) -> float:
    """Find the minimum distance between two lines in R^3.

    Parameters
    ----------
    point1 : Array
        A point on the first line.
    direction1 : Array
        The direction vector of the first line.
    point2 : Array
        A point on the second line.
    direction2 : Array
        The direction vector of the second line.

    Returns
    -------
    float
        The minimum distance between the two lines.
    """

    cross = np.cross(direction1, direction2)

    if np.isclose(np.linalg.norm(cross), 0):
        difference = point2 - point1

        distance = np.linalg.norm(
            np.cross(difference, direction1)
        ) / np.linalg.norm(direction1)

        return distance

    difference = point2 - point1

    distance = abs(difference @ cross) / np.linalg.norm(cross)

    return distance


def solve_cable_tension(
        N: int,
        L: float,
        rho: Callable[[Array], Array],
        g: float = EARTH_GRAVITY
        ) -> tuple[Array, Array]:
    """Solve for the tension in a hanging cable discretized into N segments.

    Parameters
    ----------
    N : int
        The number of segments.
    L : float
        The length of the cable.
    rho : Callable[[Array], Array]
        The density of the cable as a function of position.
    g : float, optional
        The acceleration due to gravity (default is EARTH_GRAVITY).

    Returns
    -------
    tuple[Array, Array]
        The positions and tensions along the cable.
    """
    delta_z = L / N

    z = np.linspace(0.0, 1.0, N + 1) * L

    z_mid = (z[:-1] + z[1:]) / 2

    A = (np.diag(np.ones(N)) + np.diag(-np.ones(N - 1), k=-1))

    b = rho(z_mid) * g * delta_z

    tension = np.linalg.solve(A, b.si.value)

    T = tension * u.N

    return z, T


def plot_cable_tension(z: Array, T: Array, L: float) -> Any:
    """Plot the tension along a hanging cable, colored by tension magnitude.

    Parameters
    ----------
    z : Array
        The positions along the cable.
    T : Array
        The tensions along the cable.
    L : float
        The length of the cable.

    Returns
    -------
    Any
        The matplotlib figure and axes.
    """
    plt.rcParams["text.usetex"] = False
    if hasattr(z, "value"):
        z_values = z.value
    else:
        z_values = np.asarray(z)

    if hasattr(T, "value"):
        T_values = T.value
    else:
        T_values = np.asarray(T)

    if hasattr(L, "value"):
        L_value = L.value
    else:
        L_value = L

    x = np.zeros_like(z_values)

    points = np.column_stack((x, z_values))
    segments = np.stack((points[:-1], points[1:]), axis=1)

    line = LineCollection(
        segments,
        cmap="inferno_r",
        array=T_values
    )

    fig, ax = plt.subplots()

    ax.add_collection(line)

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, L_value + 0.5)

    ax.set_xticks([])

    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label("Tension (N)")

    ax.set_ylabel("Height (m)")
    ax.set_title(
        f"Tension in a Hanging Cable with {len(T_values)} Segments"
    )
    fig.savefig("cable_tension.png", dpi=150, bbox_inches="tight")
    return fig, ax
