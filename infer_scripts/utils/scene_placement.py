"""Lightweight gravity/support placement used by ``5_construct_scene.py``.

The interface is intentionally small so a physics/REST3D-style backend can be
added later without changing scene construction or its output schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple

import numpy as np
import trimesh


OBJECT_PREFIX = "object_"


def object_index(node_id: str) -> int | None:
    if not isinstance(node_id, str) or not node_id.startswith(OBJECT_PREFIX):
        return None
    try:
        return int(node_id[len(OBJECT_PREFIX):])
    except ValueError:
        return None


def transformed_vertices(mesh: trimesh.Trimesh, transform: np.ndarray) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return vertices @ transform[:3, :3].T + transform[:3, 3]


def _translation(delta: Sequence[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(delta, dtype=np.float64)
    return matrix


def _closest_point_triangle_2d(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Closest point on a 2-D triangle, including its interior."""
    a, b, c = triangle
    v0, v1, v2 = b - a, c - a, point - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) > 1e-12:
        v = (d11 * d20 - d01 * d21) / denominator
        w = (d00 * d21 - d01 * d20) / denominator
        u = 1.0 - v - w
        if min(u, v, w) >= -1e-9:
            return point.copy()

    best = a.copy()
    best_distance = float("inf")
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        denom = edge @ edge
        fraction = 0.0 if denom < 1e-12 else float(np.clip((point - start) @ edge / denom, 0, 1))
        candidate = start + fraction * edge
        distance = float(np.sum((candidate - point) ** 2))
        if distance < best_distance:
            best, best_distance = candidate, distance
    return best


def _height_on_triangle(xz: np.ndarray, triangle: np.ndarray) -> float | None:
    projected = triangle[:, [0, 2]]
    a, b, c = projected
    matrix = np.column_stack((b - a, c - a))
    if abs(np.linalg.det(matrix)) < 1e-12:
        return None
    weights = np.linalg.solve(matrix, xz - a)
    barycentric = np.array([1.0 - weights.sum(), weights[0], weights[1]])
    return float(barycentric @ triangle[:, 1])


def find_support_surface(
    parent_mesh: trimesh.Trimesh,
    parent_transform: np.ndarray,
    query_xz: np.ndarray,
    query_y: float | None = None,
    vertical_tolerance: float = 0.05,
    normal_y_threshold: float = 0.5,
) -> Dict | None:
    """Find the closest plausible upward-facing parent support surface.

    When the child height is known, reject surfaces clearly above its bottom
    and rank the remaining candidates by their 3-D Euclidean distance to the
    query.  This prevents a marginally closer XZ projection on a much lower
    shelf or base from winning over the actual support surface.
    """
    vertices = transformed_vertices(parent_mesh, parent_transform)
    faces = np.asarray(parent_mesh.faces, dtype=np.int64)
    best = None
    for face_index, face in enumerate(faces):
        triangle = vertices[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-12 or normal[1] / norm < normal_y_threshold:
            continue
        closest_xz = _closest_point_triangle_2d(query_xz, triangle[:, [0, 2]])
        distance = float(np.linalg.norm(closest_xz - query_xz))
        height = _height_on_triangle(closest_xz, triangle)
        if height is None:
            continue
        if query_y is not None and height > query_y + vertical_tolerance:
            continue
        vertical_distance = 0.0 if query_y is None else abs(query_y - height)
        spatial_distance = float(np.hypot(distance, vertical_distance))
        candidate = {
            "face_index": int(face_index),
            "xz": closest_xz,
            "height": height,
            "horizontal_distance": distance,
            "vertical_distance": vertical_distance,
            "spatial_distance": spatial_distance,
        }
        if query_y is None:
            key = (round(distance, 9), -height)
        else:
            # Horizontal and vertical offsets share the same floor-frame units.
            # Keep horizontal distance and height only as deterministic
            # tie-breakers after the physically meaningful 3-D distance.
            key = (round(spatial_distance, 9), round(distance, 9), -height)
        if best is None or key < best[0]:
            best = (key, candidate)
    return None if best is None else best[1]


def prepare_support_graph(
    scene_graph: Dict,
    object_count: int,
    confidence_threshold: float,
) -> Tuple[List[Dict], List[int], Dict[int, List[int]]]:
    """Validate graph edges and return decisions plus supporter-first order."""
    decisions: List[Dict] = []
    parent_of: Dict[int, int | str] = {}
    child_edges: Dict[int, List[int]] = {index: [] for index in range(object_count)}

    for edge_index, edge in enumerate(scene_graph.get("edges", [])):
        decision = {
            "edge_index": edge_index,
            "child": edge.get("child"),
            "parent": edge.get("parent"),
            "relation": edge.get("relation"),
            "confidence": float(edge.get("confidence", 0.0) or 0.0),
            "operational": bool(edge.get("operational", False)),
            "status": "skipped",
            "reason": None,
        }
        child = object_index(edge.get("child"))
        parent_name = edge.get("parent")
        parent = object_index(parent_name)
        if child is None or not 0 <= child < object_count:
            decision["reason"] = "invalid_child"
        elif edge.get("relation") != "rests_on":
            decision["reason"] = "non_resting_relation"
        elif not edge.get("operational", False):
            decision["reason"] = "non_operational_edge"
        elif decision["confidence"] < confidence_threshold:
            decision["reason"] = "below_confidence_threshold"
        elif parent_name != "floor" and (parent is None or not 0 <= parent < object_count):
            decision["reason"] = "invalid_parent"
        elif child in parent_of:
            decision["reason"] = "multiple_operational_supporters"
        elif parent == child:
            decision["reason"] = "self_support"
        else:
            decision["status"] = "accepted"
            decision["reason"] = "pending_geometry"
            parent_of[child] = "floor" if parent_name == "floor" else parent
            if parent is not None:
                child_edges[parent].append(child)
        decisions.append(decision)

    # Reject cycles without allowing one bad component to poison other trees.
    accepted_by_child = {
        object_index(decision["child"]): decision
        for decision in decisions if decision["status"] == "accepted"
    }
    for start in list(parent_of):
        path: List[int] = []
        current: int | str = start
        while isinstance(current, int) and current in parent_of:
            if current in path:
                cycle = path[path.index(current):]
                for child in cycle:
                    accepted_by_child[child]["status"] = "skipped"
                    accepted_by_child[child]["reason"] = "support_cycle"
                    parent_of.pop(child, None)
                break
            path.append(current)
            current = parent_of[current]

    child_edges = {index: [] for index in range(object_count)}
    indegree = {index: 0 for index in range(object_count)}
    for child, parent in parent_of.items():
        if isinstance(parent, int):
            child_edges[parent].append(child)
            indegree[child] += 1
    queue = sorted(index for index, degree in indegree.items() if degree == 0)
    order: List[int] = []
    while queue:
        parent = queue.pop(0)
        order.append(parent)
        for child in sorted(child_edges[parent]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
                queue.sort()
    return decisions, order, child_edges


class PlacementBackend(ABC):
    name = "abstract"

    @abstractmethod
    def optimize(self, meshes, transforms, scene_graph, confidence_threshold):
        raise NotImplementedError


class GeometryPlacementBackend(PlacementBackend):
    """Deterministic support/contact adjustment without a physics runtime."""

    name = "geometry"

    def optimize(
        self,
        meshes: Sequence[trimesh.Trimesh],
        transforms: Sequence[np.ndarray],
        scene_graph: Dict,
        confidence_threshold: float,
    ) -> Tuple[List[np.ndarray], List[Dict], Dict]:
        final = [np.asarray(transform, dtype=np.float64).copy() for transform in transforms]
        decisions, order, child_edges = prepare_support_graph(
            scene_graph, len(meshes), confidence_threshold
        )
        decision_by_child = {
            object_index(decision["child"]): decision
            for decision in decisions if decision["status"] == "accepted"
        }

        def descendants(root: int) -> List[int]:
            result, stack = [], list(child_edges.get(root, []))
            while stack:
                node = stack.pop()
                result.append(node)
                stack.extend(child_edges.get(node, []))
            return result

        def move_subtree(root: int, delta: np.ndarray) -> None:
            transform = _translation(delta)
            for node in [root] + descendants(root):
                final[node] = transform @ final[node]

        parent_lookup = {}
        for decision in decisions:
            if decision["status"] != "accepted":
                continue
            child = object_index(decision["child"])
            parent_lookup[child] = decision["parent"]

        for child in order:
            if child not in parent_lookup:
                continue
            decision = decision_by_child[child]
            mesh = meshes[child]
            if mesh is None or len(mesh.vertices) == 0:
                decision.update(status="skipped", reason="missing_child_mesh")
                continue
            child_vertices = transformed_vertices(mesh, final[child])
            bottom_y = float(child_vertices[:, 1].min())
            height = max(float(np.ptp(child_vertices[:, 1])), 1e-6)
            bottom_band = child_vertices[:, 1] <= bottom_y + max(0.02 * height, 1e-5)
            query_xz = np.median(child_vertices[bottom_band][:, [0, 2]], axis=0)
            parent_name = parent_lookup[child]
            if parent_name == "floor":
                delta = np.array([0.0, -bottom_y, 0.0])
                move_subtree(child, delta)
                decision.update(
                    status="applied",
                    reason="floor_contact",
                    gap_before=bottom_y,
                    gap_after=0.0,
                    xz_shift=[0.0, 0.0],
                    translation=delta.tolist(),
                )
                continue

            parent = object_index(parent_name)
            if parent is None or meshes[parent] is None or len(meshes[parent].vertices) == 0:
                decision.update(status="skipped", reason="missing_parent_mesh")
                continue
            parent_vertices = transformed_vertices(meshes[parent], final[parent])
            parent_height = max(float(np.ptp(parent_vertices[:, 1])), 1e-6)
            surface = find_support_surface(
                meshes[parent],
                final[parent],
                query_xz,
                query_y=bottom_y,
                vertical_tolerance=max(0.05, 0.1 * height, 0.05 * parent_height),
            )
            if surface is None:
                decision.update(status="skipped", reason="no_upward_support_surface")
                continue
            parent_diagonal = float(np.linalg.norm(np.ptp(parent_vertices[:, [0, 2]], axis=0)))
            child_diagonal = float(np.linalg.norm(np.ptp(child_vertices[:, [0, 2]], axis=0)))
            shift_limit = max(0.25, parent_diagonal, child_diagonal)
            shift_xz = surface["xz"] - query_xz
            shift_distance = float(np.linalg.norm(shift_xz))
            if shift_distance > shift_limit:
                decision.update(
                    status="skipped",
                    reason="excessive_horizontal_shift",
                    proposed_xz_shift=shift_xz.tolist(),
                    horizontal_shift=shift_distance,
                    horizontal_shift_limit=shift_limit,
                )
                continue
            gap_before = bottom_y - float(surface["height"])
            delta = np.array([shift_xz[0], -gap_before, shift_xz[1]])
            move_subtree(child, delta)
            decision.update(
                status="applied",
                reason="object_contact",
                support_face=surface["face_index"],
                support_height=surface["height"],
                gap_before=gap_before,
                gap_after=0.0,
                xz_shift=shift_xz.tolist(),
                horizontal_shift=shift_distance,
                horizontal_shift_limit=shift_limit,
                translation=delta.tolist(),
            )

        intended_pairs = {
            tuple(sorted((object_index(decision["child"]), object_index(decision["parent"]))))
            for decision in decisions
            if decision["status"] == "applied" and object_index(decision["parent"]) is not None
        }
        bounds = []
        for mesh, transform in zip(meshes, final):
            vertices = transformed_vertices(mesh, transform)
            bounds.append(np.array([vertices.min(axis=0), vertices.max(axis=0)]))
        collisions = []
        for first in range(len(meshes)):
            for second in range(first + 1, len(meshes)):
                overlap = np.minimum(bounds[first][1], bounds[second][1]) - np.maximum(
                    bounds[first][0], bounds[second][0]
                )
                if np.all(overlap > 1e-6):
                    collisions.append({
                        "objects": [f"object_{first:03d}", f"object_{second:03d}"],
                        "diagnostic": "aabb_overlap",
                        "overlap_xyz": overlap.tolist(),
                        "intended_support_pair": (first, second) in intended_pairs,
                        "pose_adjusted": False,
                    })
        diagnostics = {
            "method": "transformed_aabb_broad_phase",
            "note": "diagnostic only; no unrelated object was moved to resolve collision",
            "pairs": collisions,
        }
        return final, decisions, diagnostics


def create_placement_backend(name: str) -> PlacementBackend:
    if name == "geometry":
        return GeometryPlacementBackend()
    raise ValueError(f"unsupported placement backend: {name}")
