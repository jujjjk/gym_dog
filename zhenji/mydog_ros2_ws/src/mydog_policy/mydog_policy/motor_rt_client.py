"""Single-owner local transport; no retry or HTTP fallback for motor commands."""
import socket
import time
from . import motor_rt_protocol as p


class MotorRtClient:
    def __init__(self, path='/tmp/lingzu_motor_rt.sock', timeout=.04):
        self.path, self.timeout = path, timeout
        self.sock = None
        self.sequence = 0

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def exchange(self, commands=None):
        if self.sock is None:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect(self.path)
        self.sequence = (self.sequence + 1) & 0xffffffff
        started = time.monotonic()
        try:
            self.sock.sendall(p.pack_request(
                p.OP_GET_STATE if commands is None else p.OP_COMMAND,
                self.sequence, time.monotonic_ns(),
                flags=p.FLAG_REQUIRE_TORQUE_LIMITS | p.FLAG_REQUIRE_VERIFIED_LIMITS,
                commands=() if commands is None else commands))
            data = bytearray()
            while len(data) < p.RESPONSE_SIZE:
                self.sock.settimeout(max(.0001, self.timeout - (time.monotonic() - started)))
                block = self.sock.recv(p.RESPONSE_SIZE - len(data))
                if not block:
                    raise ConnectionError('Motor RT socket closed')
                data.extend(block)
                if time.monotonic() - started > self.timeout:
                    raise TimeoutError('Motor RT response deadline exceeded')
            result = p.unpack_response(data)
            if result['sequence'] != self.sequence or result['status'] != p.STATUS_OK:
                raise RuntimeError('Motor RT response mismatch/rejected: %s' % result['status'])
            if not result['flags'] & p.RESPONSE_COMMUNICATION_OK:
                raise RuntimeError('Motor RT communication not healthy')
            ids = [s['can_id'] for s in result['states']]
            if set(ids) != {17,18,19,33,34,35,49,50,51,65,66,67} or len(set(ids)) != 12:
                raise RuntimeError('Motor RT motor IDs mismatch')
            result['payload'] = {hex(s['can_id']): s for s in result['states']}
            result['payload']['__meta__'] = dict(
                state_cache_age_ms=result['cache_age_ms'] + (time.monotonic()-started)*1000,
                # Server and client are local processes sharing CLOCK_MONOTONIC.
                # Keep the server epoch; subtracting the whole round-trip from
                # a newly created wall stamp biases the old estimate backwards.
                state_epoch_monotonic=result['timestamp_ns']/1e9-result['cache_age_ms']/1000.,
                state_server_monotonic=result['timestamp_ns']/1e9,
                communication_ok=True)
            return result
        except Exception:
            self.close()
            raise
