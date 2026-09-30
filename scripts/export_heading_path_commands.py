#!/usr/bin/env python3
"""Export curvature-aware heading-relative commands from ordered motion paths.

The exported local command is a nominal perfect-tracking preview.  The CSV also
contains the world-frame waypoint so a runtime adapter can recompute the exact
closed-loop command from the measured robot pose.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np



DEFAULT_REPRESENTATIVES = {
    "noscale__any_barrel_1": "long_nearly_straight",
    "box_85": "single_near_right_angle_turn",
    "noscale__any_barrel_12": "strong_heading_relative_lateral_component",
    "noscale__any_bin_59": "ordered_path_reversal",
    "box_45": "multi_turn_winding_path",
}

LIFT_HEIGHT_THRESHOLD_M = 0.10
LIFT_RANGE_RATIO = 0.35
LIFT_CONSECUTIVE_STEPS = 5


def wrap_to_pi(value: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(value), np.cos(value))


def yaw_from_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def rotate_world_xy_to_heading(delta_xy_w: np.ndarray, yaw_w: np.ndarray | float) -> np.ndarray:
    delta_xy_w = np.asarray(delta_xy_w, dtype=np.float64)
    yaw_w = np.asarray(yaw_w, dtype=np.float64)
    cy = np.cos(yaw_w)
    sy = np.sin(yaw_w)
    result = np.empty_like(delta_xy_w, dtype=np.float64)
    result[..., 0] = cy * delta_xy_w[..., 0] + sy * delta_xy_w[..., 1]
    result[..., 1] = -sy * delta_xy_w[..., 0] + cy * delta_xy_w[..., 1]
    return result


def _first_sustained_true(mask: np.ndarray, consecutive_steps: int, start: int = 0) -> int | None:
    run_length = 0
    for index, flag in enumerate(np.asarray(mask, dtype=bool).reshape(-1)[start:], start=start):
        run_length = run_length + 1 if bool(flag) else 0
        if run_length >= consecutive_steps:
            return index - consecutive_steps + 1
    return None


def xm0_post_pickup_window_from_rel_z(
    root_pos_w: np.ndarray,
    object_pos_w: np.ndarray,
) -> tuple[int, int, float]:
    """Match XM0's clip-derived pickup latch and persistent post-pickup command."""

    rel_z = np.asarray(object_pos_w, dtype=np.float64)[:, 2] - np.asarray(root_pos_w, dtype=np.float64)[:, 2]
    if rel_z.size == 0:
        raise ValueError("A non-empty object/root trajectory is required.")
    threshold = float(rel_z.min() + max(LIFT_HEIGHT_THRESHOLD_M, LIFT_RANGE_RATIO * np.ptp(rel_z)))
    carry_start = _first_sustained_true(rel_z >= threshold, LIFT_CONSECUTIVE_STEPS)
    if carry_start is None:
        raise ValueError("The clip never reaches the configured sustained-lift threshold.")
    # The pure-RL XM0 observation branch reads pickup_anchor_set before the
    # ordinary contact-aware carry gate.  The latch never clears in-flight,
    # even if the object is later lowered or dropped.
    return int(carry_start), int(rel_z.size), threshold


def centered_moving_average(values: np.ndarray, window_steps: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    window_steps = int(window_steps)
    if window_steps < 1 or window_steps % 2 == 0:
        raise ValueError("smoothing_steps must be a positive odd integer.")
    if window_steps == 1:
        return values.copy()
    pad = window_steps // 2
    kernel = np.full((window_steps,), 1.0 / window_steps, dtype=np.float64)
    columns = [
        np.convolve(np.pad(values[:, column], (pad, pad), mode="edge"), kernel, mode="valid")
        for column in range(values.shape[1])
    ]
    return np.stack(columns, axis=1)


def rdp_indices(points: np.ndarray, epsilon_m: float) -> list[int]:
    """Return ordered Ramer-Douglas-Peucker vertices without changing path order."""

    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return []
    if points.shape[0] <= 2:
        return list(range(points.shape[0]))

    keep = {0, points.shape[0] - 1}
    pending = [(0, points.shape[0] - 1)]
    while pending:
        start, end = pending.pop()
        candidates = points[start + 1 : end]
        if candidates.shape[0] == 0:
            continue
        segment = points[end] - points[start]
        segment_norm_sq = float(np.dot(segment, segment))
        if segment_norm_sq <= 1.0e-12:
            distances = np.linalg.norm(candidates - points[start], axis=1)
        else:
            projection = np.clip(
                ((candidates - points[start]) @ segment) / segment_norm_sq,
                0.0,
                1.0,
            )
            closest = points[start] + projection[:, None] * segment
            distances = np.linalg.norm(candidates - closest, axis=1)
        relative_index = int(np.argmax(distances))
        if float(distances[relative_index]) > float(epsilon_m):
            split = start + relative_index + 1
            keep.add(split)
            pending.extend(((start, split), (split, end)))
    return sorted(keep)


def _turn_angle_deg(incoming: np.ndarray, outgoing: np.ndarray) -> float:
    incoming_norm = float(np.linalg.norm(incoming))
    outgoing_norm = float(np.linalg.norm(outgoing))
    if min(incoming_norm, outgoing_norm) <= 1.0e-12:
        return 0.0
    cosine = float(np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def detect_path_boundaries(
    path_xy_w: np.ndarray,
    root_velocity_xy_w: np.ndarray,
    *,
    rdp_epsilon_m: float,
    hard_turn_deg: float,
    minimum_leg_m: float,
    stop_speed_mps: float,
    stop_min_steps: int,
) -> tuple[list[int], list[dict[str, Any]], list[int]]:
    """Find ordered hard corners, reversals, and sustained stop boundaries."""

    rdp_vertices = rdp_indices(path_xy_w, rdp_epsilon_m)
    hard_boundaries = {0, path_xy_w.shape[0] - 1}
    turns: list[dict[str, Any]] = []
    for vertex_offset in range(1, len(rdp_vertices) - 1):
        previous_idx, vertex_idx, next_idx = rdp_vertices[vertex_offset - 1 : vertex_offset + 2]
        incoming = path_xy_w[vertex_idx] - path_xy_w[previous_idx]
        outgoing = path_xy_w[next_idx] - path_xy_w[vertex_idx]
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        angle_deg = _turn_angle_deg(incoming, outgoing)
        is_hard = (
            angle_deg >= hard_turn_deg
            and incoming_length >= minimum_leg_m
            and outgoing_length >= minimum_leg_m
        )
        turns.append(
            {
                "frame_local": int(vertex_idx),
                "turn_angle_deg": angle_deg,
                "incoming_length_m": incoming_length,
                "outgoing_length_m": outgoing_length,
                "hard_boundary": bool(is_hard),
            }
        )
        if is_hard:
            hard_boundaries.add(int(vertex_idx))

    stationary = np.linalg.norm(np.asarray(root_velocity_xy_w, dtype=np.float64), axis=1) < stop_speed_mps
    run_start: int | None = None
    for index in range(stationary.size + 1):
        flag = bool(stationary[index]) if index < stationary.size else False
        if flag and run_start is None:
            run_start = index
        elif not flag and run_start is not None:
            if index - run_start >= stop_min_steps:
                hard_boundaries.add(max(0, min(run_start, path_xy_w.shape[0] - 1)))
                hard_boundaries.add(max(0, min(index - 1, path_xy_w.shape[0] - 1)))
            run_start = None

    boundaries = sorted(hard_boundaries)
    return boundaries, turns, rdp_vertices


def _interpolate_ordered_target(
    path_s: np.ndarray,
    path_xy_w: np.ndarray,
    path_yaw_unwrapped: np.ndarray,
    *,
    current_index: int,
    boundary_index: int,
    target_s: float,
) -> tuple[np.ndarray, float, float]:
    if target_s <= float(path_s[current_index]) + 1.0e-10:
        target_index = boundary_index if path_s[boundary_index] <= path_s[current_index] + 1.0e-10 else current_index
        return path_xy_w[target_index].copy(), float(path_yaw_unwrapped[target_index]), float(target_index)

    right = int(np.searchsorted(path_s, target_s, side="left"))
    right = max(current_index + 1, min(right, boundary_index))
    left = right - 1
    span = float(path_s[right] - path_s[left])
    alpha = 0.0 if span <= 1.0e-10 else (target_s - float(path_s[left])) / span
    alpha = max(0.0, min(1.0, float(alpha)))
    target_xy = (1.0 - alpha) * path_xy_w[left] + alpha * path_xy_w[right]
    target_yaw = (1.0 - alpha) * path_yaw_unwrapped[left] + alpha * path_yaw_unwrapped[right]
    return target_xy, float(target_yaw), float(left + alpha)


def compute_heading_path_commands(
    root_pos_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
    root_lin_vel_w: np.ndarray,
    *,
    carry_start: int,
    carry_end: int,
    lookahead_m: float,
    smoothing_steps: int,
    rdp_epsilon_m: float,
    hard_turn_deg: float,
    minimum_leg_m: float,
    stop_speed_mps: float,
    stop_min_steps: int,
) -> dict[str, Any]:
    """Compute nominal local commands and their world-frame waypoint targets."""

    root_pos_w = np.asarray(root_pos_w, dtype=np.float64)
    root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float64)
    root_lin_vel_w = np.asarray(root_lin_vel_w, dtype=np.float64)
    frame_count = int(root_pos_w.shape[0])
    if not (0 <= carry_start < carry_end <= frame_count):
        raise ValueError(f"Invalid carry window [{carry_start}, {carry_end}) for {frame_count} frames.")
    if lookahead_m <= 0.0:
        raise ValueError("lookahead_m must be positive.")

    root_yaw = yaw_from_wxyz(root_quat_wxyz)
    active_xy = centered_moving_average(root_pos_w[carry_start:carry_end, :2], smoothing_steps)
    active_yaw_unwrapped = np.unwrap(root_yaw[carry_start:carry_end])
    active_velocity_xy = root_lin_vel_w[carry_start:carry_end, :2]
    path_s = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(active_xy, axis=0), axis=1)))
    )
    boundaries, turns, rdp_vertices = detect_path_boundaries(
        active_xy,
        active_velocity_xy,
        rdp_epsilon_m=rdp_epsilon_m,
        hard_turn_deg=hard_turn_deg,
        minimum_leg_m=minimum_leg_m,
        stop_speed_mps=stop_speed_mps,
        stop_min_steps=stop_min_steps,
    )

    command = np.zeros((frame_count, 3), dtype=np.float64)
    target_xy_w = np.full((frame_count, 2), np.nan, dtype=np.float64)
    target_yaw_w = np.full((frame_count,), np.nan, dtype=np.float64)
    target_frame = np.full((frame_count,), np.nan, dtype=np.float64)
    segment_id = np.full((frame_count,), -1, dtype=np.int32)
    active = np.zeros((frame_count,), dtype=bool)
    active[carry_start:carry_end] = True

    boundary_cursor = 1
    for local_index in range(active_xy.shape[0]):
        while boundary_cursor < len(boundaries) and boundaries[boundary_cursor] <= local_index:
            boundary_cursor += 1
        next_boundary = boundaries[min(boundary_cursor, len(boundaries) - 1)]
        target_s = min(float(path_s[local_index] + lookahead_m), float(path_s[next_boundary]))
        target_xy, target_yaw, target_local_frame = _interpolate_ordered_target(
            path_s,
            active_xy,
            active_yaw_unwrapped,
            current_index=local_index,
            boundary_index=next_boundary,
            target_s=target_s,
        )
        frame = carry_start + local_index
        delta_xy_w = target_xy - root_pos_w[frame, :2]
        command[frame, :2] = rotate_world_xy_to_heading(delta_xy_w, root_yaw[frame])
        command[frame, 2] = float(wrap_to_pi(target_yaw - root_yaw[frame]))
        target_xy_w[frame] = target_xy
        target_yaw_w[frame] = float(wrap_to_pi(target_yaw))
        target_frame[frame] = carry_start + target_local_frame
        segment_id[frame] = max(0, boundary_cursor - 1)

    hard_boundary_frames = [carry_start + index for index in boundaries]
    for turn in turns:
        turn["frame"] = carry_start + int(turn["frame_local"])

    return {
        "command": command,
        "active": active,
        "target_xy_w": target_xy_w,
        "target_yaw_w": target_yaw_w,
        "target_frame": target_frame,
        "segment_id": segment_id,
        "root_yaw_w": root_yaw,
        "smoothed_active_xy_w": active_xy,
        "active_path_s": path_s,
        "hard_boundary_frames": np.asarray(hard_boundary_frames, dtype=np.int32),
        "rdp_vertex_frames": np.asarray([carry_start + index for index in rdp_vertices], dtype=np.int32),
        "turns": turns,
        "path_arc_length_m": float(path_s[-1]),
        "path_net_displacement_m": float(np.linalg.norm(active_xy[-1] - active_xy[0])),
    }


def prune_short_polyline_legs(
    path_xy_w: np.ndarray,
    vertex_indices: list[int],
    minimum_leg_m: float,
) -> list[int]:
    """Drop RDP vertices that create sub-threshold command legs."""

    path_xy_w = np.asarray(path_xy_w, dtype=np.float64)
    if path_xy_w.ndim != 2 or path_xy_w.shape[1] != 2:
        raise ValueError(f"path_xy_w must have shape (T, 2), got {path_xy_w.shape}.")
    if not math.isfinite(minimum_leg_m) or minimum_leg_m < 0.0:
        raise ValueError("minimum_leg_m must be finite and non-negative.")
    vertices = [int(index) for index in vertex_indices]
    if not vertices:
        return []
    if vertices != sorted(set(vertices)):
        raise ValueError("vertex_indices must be strictly increasing and unique.")
    if vertices[0] < 0 or vertices[-1] >= path_xy_w.shape[0]:
        raise ValueError("vertex_indices contain an out-of-range path index.")

    # Repeatedly remove one endpoint of the shortest bad leg.  Endpoints of
    # the full path are immutable; for an interior bad leg, remove the vertex
    # whose deletion produces the smaller geometric deviation.
    while len(vertices) > 2:
        points = path_xy_w[vertices]
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        bad = np.flatnonzero(lengths < minimum_leg_m)
        if bad.size == 0:
            break
        leg = int(bad[np.argmin(lengths[bad])])
        if leg == 0:
            remove_at = 1
        elif leg == len(vertices) - 2:
            remove_at = len(vertices) - 2
        else:
            left = points[leg - 1]
            first = points[leg]
            second = points[leg + 1]
            right = points[leg + 2]
            remove_first_cost = _point_segment_distance(first, left, second)
            remove_second_cost = _point_segment_distance(second, first, right)
            remove_at = leg if remove_first_cost <= remove_second_cost else leg + 1
        del vertices[remove_at]
    return vertices


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    norm_sq = float(np.dot(segment, segment))
    if norm_sq <= 1.0e-12:
        return float(np.linalg.norm(point - start))
    alpha = float(np.clip(np.dot(point - start, segment) / norm_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + alpha * segment)))


def _turn_window_steps(
    angle_rad: float,
    *,
    minimum_steps: int,
    maximum_steps: int,
) -> int:
    fraction_of_half_turn = min(1.0, abs(float(angle_rad)) / math.pi)
    proportional = int(round(fraction_of_half_turn * maximum_steps))
    return max(minimum_steps, min(maximum_steps, proportional))


def compute_turn_then_forward_commands(
    root_pos_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
    *,
    carry_start: int,
    carry_end: int,
    smoothing_steps: int = 5,
    rdp_epsilon_m: float = 0.06,
    minimum_leg_m: float = 0.10,
    minimum_turn_deg: float = 25.0,
    forward_command_m: float = 0.15,
    minimum_forward_command_m: float = 1.0e-4,
    minimum_turn_steps: int = 8,
    maximum_turn_steps: int = 30,
    minimum_forward_steps_between_turns: int = 5,
) -> dict[str, Any]:
    """Precompute mutually exclusive zero, forward, and yaw-only commands.

    The result is an open-loop actor input indexed only by reference-motion
    time.  It deliberately does not use the simulated robot pose.  Runtime
    still gates these rows with the existing pickup latch.
    """

    root_pos_w = np.asarray(root_pos_w, dtype=np.float64)
    root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float64)
    if root_pos_w.ndim != 2 or root_pos_w.shape[1] < 2:
        raise ValueError(f"root_pos_w must have shape (T, >=2), got {root_pos_w.shape}.")
    frame_count = int(root_pos_w.shape[0])
    if root_quat_wxyz.shape != (frame_count, 4):
        raise ValueError(
            f"root_quat_wxyz must have shape ({frame_count}, 4), got {root_quat_wxyz.shape}."
        )
    if not (0 <= carry_start < carry_end <= frame_count):
        raise ValueError(f"Invalid carry window [{carry_start}, {carry_end}) for {frame_count} frames.")
    finite_scalars = {
        "rdp_epsilon_m": rdp_epsilon_m,
        "minimum_leg_m": minimum_leg_m,
        "minimum_turn_deg": minimum_turn_deg,
        "forward_command_m": forward_command_m,
        "minimum_forward_command_m": minimum_forward_command_m,
    }
    for name, value in finite_scalars.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}.")
    if forward_command_m <= 0.0:
        raise ValueError("forward_command_m must be positive.")
    if minimum_forward_command_m >= forward_command_m:
        raise ValueError("minimum_forward_command_m must be smaller than forward_command_m.")
    if minimum_turn_deg > 180.0:
        raise ValueError("minimum_turn_deg must not exceed 180 degrees.")
    integer_values = {
        "minimum_turn_steps": minimum_turn_steps,
        "maximum_turn_steps": maximum_turn_steps,
        "minimum_forward_steps_between_turns": minimum_forward_steps_between_turns,
    }
    for name, value in integer_values.items():
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if minimum_turn_steps > maximum_turn_steps:
        raise ValueError("minimum_turn_steps must not exceed maximum_turn_steps.")

    active_xy = centered_moving_average(root_pos_w[carry_start:carry_end, :2], smoothing_steps)
    if active_xy.shape[0] < 2:
        raise ValueError("The post-pickup command window must contain at least two frames.")
    raw_vertices = rdp_indices(active_xy, rdp_epsilon_m)
    vertices = prune_short_polyline_legs(active_xy, raw_vertices, minimum_leg_m)
    if len(vertices) < 2:
        raise ValueError("Path simplification did not retain both path endpoints.")

    command = np.zeros((frame_count, 3), dtype=np.float64)
    phase = np.zeros((frame_count,), dtype=np.uint8)
    segment_id = np.full((frame_count,), -1, dtype=np.int32)
    leg_records: list[dict[str, Any]] = []
    leg_headings: list[float] = []

    for leg_id, (start_local, end_local) in enumerate(zip(vertices[:-1], vertices[1:])):
        delta = active_xy[end_local] - active_xy[start_local]
        length_m = float(np.linalg.norm(delta))
        if length_m <= 1.0e-12:
            raise ValueError(f"Simplified path contains a zero-length leg at index {leg_id}.")
        heading = float(math.atan2(delta[1], delta[0]))
        leg_headings.append(heading)
        start_frame = carry_start + start_local
        end_frame = carry_start + end_local
        for frame in range(start_frame, end_frame):
            remaining_m = float(np.linalg.norm(active_xy[end_local] - active_xy[frame - carry_start]))
            dx = min(forward_command_m, remaining_m)
            if dx >= minimum_forward_command_m:
                command[frame, 0] = dx
                phase[frame] = 1
            segment_id[frame] = leg_id
        leg_records.append(
            {
                "leg_id": leg_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "length_m": length_m,
                "heading_w_rad": heading,
            }
        )
    segment_id[carry_start + vertices[-1]] = len(vertices) - 2

    turn_records: list[dict[str, Any]] = []

    def install_turn(*, vertex_offset: int, frame: int, angle_rad: float, kind: str) -> None:
        if abs(math.degrees(angle_rad)) < minimum_turn_deg:
            return
        requested_steps = _turn_window_steps(
            angle_rad,
            minimum_steps=minimum_turn_steps,
            maximum_steps=maximum_turn_steps,
        )
        if kind == "initial_alignment":
            window_start = carry_start
            next_boundary = carry_start + vertices[1]
            latest_end = next_boundary - minimum_forward_steps_between_turns
            window_end = min(carry_start + requested_steps, latest_end)
        else:
            previous_boundary = carry_start + vertices[vertex_offset - 1]
            earliest_start = previous_boundary + minimum_forward_steps_between_turns
            window_end = frame
            window_start = max(earliest_start, frame - requested_steps)
        if window_end <= window_start:
            raise ValueError(
                f"No non-overlapping yaw window is available for {kind} at frame {frame}; "
                "increase temporal path separation or reduce the turn threshold."
            )
        command[window_start:window_end] = 0.0
        command[window_start:window_end, 2] = angle_rad
        phase[window_start:window_end] = 2
        turn_records.append(
            {
                "kind": kind,
                "vertex_frame": frame,
                "window_start": window_start,
                "window_end": window_end,
                "window_steps": window_end - window_start,
                "requested_window_steps": requested_steps,
                "turn_angle_rad": angle_rad,
                "turn_angle_deg": math.degrees(angle_rad),
            }
        )

    root_yaw = yaw_from_wxyz(root_quat_wxyz)
    initial_angle = float(wrap_to_pi(leg_headings[0] - root_yaw[carry_start]))
    install_turn(
        vertex_offset=0,
        frame=carry_start,
        angle_rad=initial_angle,
        kind="initial_alignment",
    )
    for vertex_offset in range(1, len(vertices) - 1):
        angle = float(wrap_to_pi(leg_headings[vertex_offset] - leg_headings[vertex_offset - 1]))
        install_turn(
            vertex_offset=vertex_offset,
            frame=carry_start + vertices[vertex_offset],
            angle_rad=angle,
            kind="path_corner",
        )

    tolerance = 1.0e-12
    if np.any(np.abs(command[:, 1]) > tolerance):
        raise AssertionError("Generated command unexpectedly contains lateral motion.")
    if np.any((np.abs(command[:, 0]) > tolerance) & (np.abs(command[:, 2]) > tolerance)):
        raise AssertionError("Generated command unexpectedly couples forward and yaw components.")
    if np.any(command[:, 0] < -tolerance):
        raise AssertionError("Generated forward command must never be negative.")
    if np.any(command[:carry_start]) or np.any(command[carry_end:]):
        raise AssertionError("Generated command escaped the configured post-pickup window.")
    if np.any((phase == 0) & (np.abs(command).max(axis=1) > tolerance)):
        raise AssertionError("Zero-phase rows must contain zero commands.")
    if np.any((phase == 1) & (command[:, 0] <= tolerance)):
        raise AssertionError("Forward-phase rows must contain a positive forward command.")
    if np.any((phase == 2) & (np.abs(command[:, 2]) <= tolerance)):
        raise AssertionError("Yaw-phase rows must contain a non-zero yaw command.")

    return {
        "command": command,
        "phase": phase,
        "segment_id": segment_id,
        "smoothed_active_xy_w": active_xy,
        "raw_rdp_vertex_frames": np.asarray(
            [carry_start + index for index in raw_vertices], dtype=np.int32
        ),
        "vertex_frames": np.asarray([carry_start + index for index in vertices], dtype=np.int32),
        "legs": leg_records,
        "turn_windows": turn_records,
        "carry_window": [carry_start, carry_end],
        "phase_counts": np.bincount(phase.astype(np.int64), minlength=3),
        "path_arc_length_m": float(
            sum(record["length_m"] for record in leg_records)
        ),
        "path_net_displacement_m": float(np.linalg.norm(active_xy[-1] - active_xy[0])),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_clip(path: Path) -> dict[str, np.ndarray]:
    required = {"body_pos_w", "body_quat_w", "body_lin_vel_w", "object_pos_w", "fps"}
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")
        return {key: np.asarray(data[key]).copy() for key in required}


def _write_clip_csv(
    path: Path,
    *,
    root_pos_w: np.ndarray,
    fps: float,
    result: dict[str, Any],
) -> None:
    fields = [
        "frame",
        "time_s",
        "active",
        "segment_id",
        "root_x_w",
        "root_y_w",
        "root_yaw_w_rad",
        "target_x_w",
        "target_y_w",
        "target_yaw_w_rad",
        "target_frame",
        "command_dx_heading_m",
        "command_dy_heading_m",
        "command_dyaw_rad",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for frame in range(root_pos_w.shape[0]):
            target_xy = result["target_xy_w"][frame]
            writer.writerow(
                {
                    "frame": frame,
                    "time_s": frame / fps,
                    "active": int(result["active"][frame]),
                    "segment_id": int(result["segment_id"][frame]),
                    "root_x_w": float(root_pos_w[frame, 0]),
                    "root_y_w": float(root_pos_w[frame, 1]),
                    "root_yaw_w_rad": float(result["root_yaw_w"][frame]),
                    "target_x_w": "" if not np.isfinite(target_xy[0]) else float(target_xy[0]),
                    "target_y_w": "" if not np.isfinite(target_xy[1]) else float(target_xy[1]),
                    "target_yaw_w_rad": ""
                    if not np.isfinite(result["target_yaw_w"][frame])
                    else float(result["target_yaw_w"][frame]),
                    "target_frame": ""
                    if not np.isfinite(result["target_frame"][frame])
                    else float(result["target_frame"][frame]),
                    "command_dx_heading_m": float(result["command"][frame, 0]),
                    "command_dy_heading_m": float(result["command"][frame, 1]),
                    "command_dyaw_rad": float(result["command"][frame, 2]),
                }
            )


def _plot_clip(
    path: Path,
    *,
    clip_id: str,
    root_pos_w: np.ndarray,
    fps: float,
    carry_start: int,
    carry_end: int,
    result: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    command = result["command"]
    active_frames = np.flatnonzero(result["active"])
    time_axis = np.arange(root_pos_w.shape[0]) / fps
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=160)

    path_ax = axes[0]
    path_ax.plot(root_pos_w[carry_start:carry_end, 0], root_pos_w[carry_start:carry_end, 1], color="#1769aa")
    path_ax.scatter(root_pos_w[carry_start, 0], root_pos_w[carry_start, 1], color="#2e7d32", label="command start")
    path_ax.scatter(
        root_pos_w[carry_end - 1, 0],
        root_pos_w[carry_end - 1, 1],
        color="#c62828",
        label="command end",
    )
    for boundary in result["hard_boundary_frames"]:
        path_ax.scatter(root_pos_w[boundary, 0], root_pos_w[boundary, 1], color="#6a1b9a", marker="D", s=28)
    stride = max(1, active_frames.size // 12)
    arrow_frames = active_frames[::stride]
    valid_targets = result["target_xy_w"][arrow_frames]
    path_ax.quiver(
        root_pos_w[arrow_frames, 0],
        root_pos_w[arrow_frames, 1],
        valid_targets[:, 0] - root_pos_w[arrow_frames, 0],
        valid_targets[:, 1] - root_pos_w[arrow_frames, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#ef6c00",
        width=0.006,
    )
    path_ax.set_title(f"{clip_id}: ordered world path and waypoints")
    path_ax.set_aspect("equal", adjustable="datalim")
    path_ax.grid(alpha=0.25)
    path_ax.legend(fontsize=8)

    axes[1].plot(time_axis, command[:, 0], label="dx", color="#1565c0")
    axes[1].plot(time_axis, command[:, 1], label="dy", color="#2e7d32")
    axes[1].axvspan(carry_start / fps, carry_end / fps, color="#90caf9", alpha=0.15)
    axes[1].set_title("Nominal current-heading XY command")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("metres")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(time_axis, np.rad2deg(command[:, 2]), color="#6a1b9a", label="dyaw")
    axes[2].plot(time_axis, np.linalg.norm(command[:, :2], axis=1) * 100.0, color="#ef6c00", label="xy norm (cm)")
    axes[2].axvspan(carry_start / fps, carry_end / fps, color="#90caf9", alpha=0.15)
    axes[2].set_title("Yaw preview and XY magnitude")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _sample_frames(carry_start: int, carry_end: int, boundaries: np.ndarray) -> list[int]:
    span = carry_end - carry_start
    frames = {
        carry_start,
        carry_start + span // 4,
        carry_start + span // 2,
        carry_start + 3 * span // 4,
        carry_end - 1,
    }
    for boundary in boundaries[1:-1]:
        frames.update((int(boundary) - 5, int(boundary), int(boundary) + 5))
    return sorted(frame for frame in frames if carry_start <= frame < carry_end)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clip", action="append", default=[])
    parser.add_argument("--lookahead-m", type=float, default=0.15)
    parser.add_argument("--smoothing-steps", type=int, default=5)
    parser.add_argument("--rdp-epsilon-m", type=float, default=0.04)
    parser.add_argument("--hard-turn-deg", type=float, default=60.0)
    parser.add_argument("--minimum-leg-m", type=float, default=0.08)
    parser.add_argument("--stop-speed-mps", type=float, default=0.03)
    parser.add_argument("--stop-min-steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_dir = args.motion_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    clip_reasons = dict(DEFAULT_REPRESENTATIVES)
    clip_ids = list(args.clip) if args.clip else list(DEFAULT_REPRESENTATIVES)
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("Duplicate --clip values are not allowed.")

    map_path = motion_dir / "_clip_object_urdf_map.json"
    if not map_path.is_file():
        raise FileNotFoundError(f"Required object map is missing: {map_path}")

    manifest_clips: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for clip_id in clip_ids:
        source_path = motion_dir / f"{clip_id}.npz"
        if not source_path.is_file():
            raise FileNotFoundError(f"Requested clip does not exist: {source_path}")
        data = _load_clip(source_path)
        root_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)[:, 0]
        root_quat_wxyz = np.asarray(data["body_quat_w"], dtype=np.float64)[:, 0]
        root_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float64)[:, 0]
        object_pos_w = np.asarray(data["object_pos_w"], dtype=np.float64)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        carry_start, carry_end, lift_threshold = xm0_post_pickup_window_from_rel_z(root_pos_w, object_pos_w)
        result = compute_heading_path_commands(
            root_pos_w,
            root_quat_wxyz,
            root_lin_vel_w,
            carry_start=carry_start,
            carry_end=carry_end,
            lookahead_m=args.lookahead_m,
            smoothing_steps=args.smoothing_steps,
            rdp_epsilon_m=args.rdp_epsilon_m,
            hard_turn_deg=args.hard_turn_deg,
            minimum_leg_m=args.minimum_leg_m,
            stop_speed_mps=args.stop_speed_mps,
            stop_min_steps=args.stop_min_steps,
        )

        csv_path = out_dir / f"{clip_id}__heading_path_command.csv"
        npz_path = out_dir / f"{clip_id}__heading_path_command.npz"
        plot_path = out_dir / f"{clip_id}__heading_path_command.png"
        _write_clip_csv(csv_path, root_pos_w=root_pos_w, fps=fps, result=result)
        np.savez_compressed(
            npz_path,
            command=result["command"].astype(np.float32),
            active=result["active"],
            target_xy_w=result["target_xy_w"].astype(np.float32),
            target_yaw_w=result["target_yaw_w"].astype(np.float32),
            target_frame=result["target_frame"].astype(np.float32),
            segment_id=result["segment_id"],
            hard_boundary_frames=result["hard_boundary_frames"],
            carry_window=np.asarray([carry_start, carry_end], dtype=np.int32),
            fps=np.asarray([fps], dtype=np.float32),
        )
        _plot_clip(
            plot_path,
            clip_id=clip_id,
            root_pos_w=root_pos_w,
            fps=fps,
            carry_start=carry_start,
            carry_end=carry_end,
            result=result,
        )

        for frame in _sample_frames(carry_start, carry_end, result["hard_boundary_frames"]):
            sample_rows.append(
                {
                    "clip_id": clip_id,
                    "selection_reason": clip_reasons.get(clip_id, "user_selected"),
                    "frame": frame,
                    "time_s": frame / fps,
                    "segment_id": int(result["segment_id"][frame]),
                    "dx_heading_m": float(result["command"][frame, 0]),
                    "dy_heading_m": float(result["command"][frame, 1]),
                    "dyaw_rad": float(result["command"][frame, 2]),
                    "dyaw_deg": float(np.rad2deg(result["command"][frame, 2])),
                    "target_x_w": float(result["target_xy_w"][frame, 0]),
                    "target_y_w": float(result["target_xy_w"][frame, 1]),
                }
            )

        net = float(result["path_net_displacement_m"])
        arc = float(result["path_arc_length_m"])
        manifest_clips.append(
            {
                "clip_id": clip_id,
                "selection_reason": clip_reasons.get(clip_id, "user_selected"),
                "source_path": str(source_path),
                "source_sha256": _sha256(source_path),
                "fps": fps,
                "frame_count": int(root_pos_w.shape[0]),
                "carry_window": [carry_start, carry_end],
                "pickup_threshold_rel_z_m": lift_threshold,
                "path_arc_length_m": arc,
                "path_net_displacement_m": net,
                "path_arc_to_net_ratio": arc / max(net, 1.0e-9),
                "hard_boundary_frames": result["hard_boundary_frames"].astype(int).tolist(),
                "turns": result["turns"],
                "outputs": {
                    "csv": csv_path.name,
                    "npz": npz_path.name,
                    "plot": plot_path.name,
                },
            }
        )

    sample_path = out_dir / "representative_command_samples.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)

    manifest = {
        "created_unix_time": time.time(),
        "semantics": "nominal_perfect_tracking_current_robot_heading_relative_pose",
        "closed_loop_runtime_rule": (
            "Recompute R(-robot_yaw) @ (target_xy_w - robot_xy_w) and "
            "wrap(target_yaw_w - robot_yaw) from the exported waypoint."
        ),
        "command_units": ["metre", "metre", "radian"],
        "motion_dir": str(motion_dir),
        "object_map_sha256": _sha256(map_path),
        "algorithm": {
            "lookahead_m": args.lookahead_m,
            "smoothing_steps": args.smoothing_steps,
            "rdp_epsilon_m": args.rdp_epsilon_m,
            "hard_turn_deg": args.hard_turn_deg,
            "minimum_leg_m": args.minimum_leg_m,
            "stop_speed_mps": args.stop_speed_mps,
            "stop_min_steps": args.stop_min_steps,
            "activation_window": (
                "XM0-compatible clip-derived rel_z pickup latch; once active, "
                "the command remains active through the source clip end"
            ),
            "hard_boundaries": "ordered RDP sharp turns/reversals plus sustained-stop edges",
            "boundary_rule": "lookahead never crosses the next hard boundary",
        },
        "sample_csv": sample_path.name,
        "clips": manifest_clips,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output_dir={out_dir}")
    print(f"manifest={manifest_path}")
    print(f"clips={len(manifest_clips)}")


if __name__ == "__main__":
    main()
