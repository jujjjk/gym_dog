"""Local binary socket server kept beside the FastAPI motor service."""

from __future__ import annotations

import os
import socket
import threading
import time

from motor_rt_protocol import (
    COMMAND_ENTRY,
    FLAG_ENABLE_FIRST,
    FLAG_REQUIRE_TORQUE_LIMITS,
    FLAG_REQUIRE_VERIFIED_LIMITS,
    FLAG_STOP_FIRST,
    MOTOR_COUNT,
    OP_COMMAND,
    OP_GET_STATE,
    OP_STOP,
    REQUEST_HEADER,
    STATUS_BAD_REQUEST,
    STATUS_IO_ERROR,
    STATUS_OK,
    STATUS_SAFETY_REJECTED,
    pack_response,
    unpack_commands,
    unpack_request_header,
)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        block = sock.recv(size - len(result))
        if not block:
            raise ConnectionError("client disconnected")
        result.extend(block)
    return bytes(result)


class MotorRtServer:
    """Serve state reads and the sole high-rate command endpoint."""

    def __init__(self, socket_path, *, get_state, send_motion, stop_motors):
        self.socket_path = str(socket_path)
        self.get_state = get_state
        self.send_motion = send_motion
        self.stop_motors = stop_motors
        self._stop = threading.Event()
        self._listener = None
        self._thread = None
        self._clients = set()
        self._clients_lock = threading.Lock()
        self._command_lock = threading.Lock()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(self.socket_path)
        os.chmod(self.socket_path, 0o660)
        listener.listen(4)
        listener.settimeout(0.2)
        self._listener = listener
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._accept_loop,
            name="motor-rt-socket",
            daemon=True,
        )
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        with self._clients_lock:
            clients = tuple(self._clients)
        for client in clients:
            try:
                client.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                client, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client.settimeout(0.5)
            with self._clients_lock:
                self._clients.add(client)
            threading.Thread(
                target=self._client_loop,
                args=(client,),
                name="motor-rt-client",
                daemon=True,
            ).start()

    def _client_loop(self, client):
        try:
            while not self._stop.is_set():
                header_payload = _recv_exact(client, REQUEST_HEADER.size)
                header = unpack_request_header(header_payload)
                commands = ()
                if header["opcode"] == OP_COMMAND:
                    commands = unpack_commands(
                        _recv_exact(client, MOTOR_COUNT * COMMAND_ENTRY.size)
                    )
                response = self._dispatch(header, commands)
                client.sendall(response)
        except (ConnectionError, OSError, ValueError):
            pass
        finally:
            with self._clients_lock:
                self._clients.discard(client)
            try:
                client.close()
            except OSError:
                pass

    def _dispatch(self, header, commands):
        status = STATUS_OK
        timing = {}
        try:
            if header["opcode"] == OP_COMMAND:
                flags = header["flags"]
                with self._command_lock:
                    timing = self.send_motion(
                        commands,
                        enable_first=bool(flags & FLAG_ENABLE_FIRST),
                        stop_first=bool(flags & FLAG_STOP_FIRST),
                        require_hardware_torque_limits=bool(
                            flags & FLAG_REQUIRE_TORQUE_LIMITS
                        ),
                        require_verified_hardware_safety_limits=bool(
                            flags & FLAG_REQUIRE_VERIFIED_LIMITS
                        ),
                    )
            elif header["opcode"] == OP_STOP:
                with self._command_lock:
                    self.stop_motors()
            elif header["opcode"] != OP_GET_STATE:
                status = STATUS_BAD_REQUEST
        except PermissionError:
            status = STATUS_SAFETY_REJECTED
        except Exception as exc:
            status = STATUS_IO_ERROR
            print(f"[RT_SOCKET] request failed: {exc}")

        state = self.get_state()
        return pack_response(
            status=status,
            flags=state["flags"],
            sequence=header["sequence"],
            cache_sequence=state["cache_sequence"],
            timestamp_ns=state["timestamp_ns"],
            cache_age_ms=state["cache_age_ms"],
            spi_ms=float(timing.get("spi_send_ms", 0.0)),
            total_ms=float(timing.get("total_ms", 0.0)),
            board_a_seq=state["board_a_seq"],
            board_b_seq=state["board_b_seq"],
            states=state["states"],
        )
