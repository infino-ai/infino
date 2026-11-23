# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Infino Authors
"""A hosted connection that gets no usable answer raises `UnavailableError`.

A stub endpoint on a loopback port stands in for the hosted service, so these
run with no real endpoint.
"""

import socket
import threading
import time

import infino
import pytest


def _stub(answer):
    """Serve one connection on a loopback port: read the request, then call
    `answer(conn)`. Returns the connect URL and the server thread."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def serve():
        conn, _ = listener.accept()
        with conn, listener:
            conn.recv(65536)
            answer(conn)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{port}/db", thread


def test_a_503_raises_unavailable_error():
    def answer(conn):
        body = b'{"error":"try later"}'
        conn.sendall(
            b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\nConnection: close\r\n\r\n%s" % (len(body), body)
        )

    url, thread = _stub(answer)
    db = infino.connect(url, api_key="ik_test")
    with pytest.raises(infino.UnavailableError, match="503"):
        db.list_tables()
    thread.join()


def test_request_timeout_ends_a_silent_request():
    url, thread = _stub(lambda conn: time.sleep(5))
    db = infino.connect(url, api_key="ik_test", request_timeout=0.3)
    start = time.monotonic()
    with pytest.raises(infino.UnavailableError):
        db.list_tables()
    assert time.monotonic() - start < 4
    # An InfinoError, so a broad handler catches it too.
    assert issubclass(infino.UnavailableError, infino.InfinoError)


@pytest.mark.parametrize("bad", [0, -1, float("nan")])
def test_request_timeout_must_be_positive(bad):
    with pytest.raises(ValueError):
        infino.connect("http://127.0.0.1:1/db", api_key="ik_test", request_timeout=bad)
