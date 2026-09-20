import pytest
from mydog_policy import motor_rt_client as m
from mydog_policy import motor_rt_protocol as p


@pytest.fixture(autouse=True)
def mocked_unix_socket_family(monkeypatch):
    # These tests replace socket creation entirely; allow running on Windows
    # Python builds without AF_UNIX without changing the Linux runtime client.
    if not hasattr(m.socket, 'AF_UNIX'):
        monkeypatch.setattr(m.socket, 'AF_UNIX', object(), raising=False)


class FakeSocket:
    def __init__(self, reject=False):
        self.reject = reject
        self.closed = False
    def settimeout(self, value): pass
    def connect(self, path): pass
    def close(self): self.closed = True
    def sendall(self, data):
        self.header = p.unpack_request_header(data[:p.REQUEST_HEADER.size])
        self.reply = p.pack_response(
            status=p.STATUS_SAFETY_REJECTED if self.reject else p.STATUS_OK,
            flags=p.RESPONSE_COMMUNICATION_OK, sequence=self.header['sequence'],
            cache_sequence=1, timestamp_ns=1, cache_age_ms=1., spi_ms=1., total_ms=2.,
            board_a_seq=1, board_b_seq=1,
            states=[dict(can_id=i, online=True) for i in (17,18,19,33,34,35,49,50,51,65,66,67)])
    def recv(self, size):
        part, self.reply = self.reply[:min(size,17)], self.reply[min(size,17):]
        return part


def test_fragmented_reply_and_no_enable_or_stop_flags(monkeypatch):
    sock = FakeSocket()
    monkeypatch.setattr(m.socket, 'socket', lambda *a: sock)
    client = m.MotorRtClient(timeout=1.)
    commands = [p.MotorCommand(i,0,0,0,40,1) for i in (17,18,19,33,34,35,49,50,51,65,66,67)]
    result = client.exchange(commands)
    assert sock.header['flags'] == p.FLAG_REQUIRE_TORQUE_LIMITS | p.FLAG_REQUIRE_VERIFIED_LIMITS
    assert len(result['states']) == 12
    client.exchange()
    assert sock.header['opcode'] == p.OP_GET_STATE
    client.close()
    assert sock.closed


def test_rejection_closes_connection_without_retry(monkeypatch):
    sock = FakeSocket(reject=True)
    monkeypatch.setattr(m.socket, 'socket', lambda *a: sock)
    client = m.MotorRtClient()
    with pytest.raises(RuntimeError): client.exchange()
    assert sock.closed and client.sock is None
    assert client.sequence == 1
