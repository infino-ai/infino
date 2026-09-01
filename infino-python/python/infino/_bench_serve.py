"""EXPERIMENTAL benchmark serve mode CLI — **not a production server**.

Serves a single table over a minimal raw-TCP wire so a networked benchmark
client (e.g. VectorDBBench) can drive the embedded engine on a dedicated host.
No auth, no TLS, no durability. The supported infino interface is embedded
(``connect``/``open_table``); this exists purely for throughput benchmarking.
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(
        prog="infino-bench-serve",
        description=(
            "EXPERIMENTAL infino benchmark serve mode (raw-TCP). No auth/TLS/durability; "
            "NOT production-validated."
        ),
    )
    p.add_argument("--data", required=True, help="data path (the connect uri)")
    p.add_argument("--table", default="vdbbench_infino", help="table name to serve")
    p.add_argument("--col", default="emb", help="vector column name")
    p.add_argument(
        "--addr",
        default="127.0.0.1:50052",
        help="bind address; use 0.0.0.0:PORT to accept remote clients",
    )
    p.add_argument("--cache-bytes", type=int, default=21_474_836_480, help="block-cache budget")
    p.add_argument(
        "--id-col",
        default="",
        help=(
            "scalar int64 column to project and return as the result id "
            "(dataset id, 8-byte LE); empty returns the engine _id (16-byte)"
        ),
    )
    a = p.parse_args()

    from infino._infino import bench_serve_tcp

    bench_serve_tcp(a.data, a.table, a.col, a.addr, a.cache_bytes, a.id_col)


if __name__ == "__main__":
    main()
