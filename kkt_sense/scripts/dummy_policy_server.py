"""Dummy websocket policy server for KKT smoke tests."""

from __future__ import annotations

import argparse
import logging
import math
import time
from typing import List
from typing import Optional

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.server import serve


def _parse_constant_action(raw: str, action_dim: int) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != action_dim:
        raise ValueError(
            "constant-action length {} does not match action-dim {}".format(
                len(parts),
                action_dim,
            )
        )
    return [float(p) for p in parts]


def _build_actions(
    mode: str,
    chunk_size: int,
    action_dim: int,
    constant_action: Optional[List[float]],
) -> np.ndarray:
    if mode == "zero":
        return np.zeros((chunk_size, action_dim), dtype=np.float32)

    if mode == "constant":
        if constant_action is None:
            raise ValueError("constant_action must be provided when action_mode='constant'")
        return np.tile(np.asarray(constant_action, dtype=np.float32), (chunk_size, 1))

    if mode == "sine":
        t = time.time()
        actions = np.zeros((chunk_size, action_dim), dtype=np.float32)
        actions[:, 0] = 0.01 * math.sin(t)
        if action_dim > 1:
            actions[:, 1] = 0.01 * math.cos(t)
        return actions

    raise ValueError("Unsupported action mode: {}".format(mode))


def _handle_connection(websocket, packer, args, constant_action):
    logging.info("Client connected")

    metadata = {
        "dummy_policy": True,
        "chunk_size": args.chunk_size,
        "action_dim": args.action_dim,
        "action_mode": args.action_mode,
    }
    websocket.send(packer.pack(metadata))

    for message in websocket:
        _ = msgpack_numpy.unpackb(message)
        actions = _build_actions(
            args.action_mode,
            args.chunk_size,
            args.action_dim,
            constant_action,
        )
        response = {
            "actions": actions,
            "server_timing": {"dummy": True},
        }
        websocket.send(packer.pack(response))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy websocket policy server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--action-mode", default="zero", choices=["zero", "constant", "sine"])
    parser.add_argument("--constant-action", default="0,0,0,0,0,0,0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    constant_action = None
    if args.action_mode == "constant":
        constant_action = _parse_constant_action(args.constant_action, args.action_dim)

    logging.info(
        "Starting dummy policy server on %s:%s (chunk_size=%s, action_dim=%s, action_mode=%s)",
        args.host,
        args.port,
        args.chunk_size,
        args.action_dim,
        args.action_mode,
    )

    packer = msgpack_numpy.Packer()

    with serve(
        lambda ws: _handle_connection(ws, packer, args, constant_action),
        args.host,
        args.port,
        compression=None,
        max_size=None,
    ) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()