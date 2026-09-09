import time

import pytest

modal_grid = pytest.importorskip("modal_grid",
                                 reason="the Modal runner needs the `modal` package")
_drain = modal_grid._drain_with_stall_timeout


def test_a_healthy_stream_passes_straight_through():
    assert list(_drain(iter([1, 2, 3]), timeout_s=5)) == [1, 2, 3]


def test_a_slow_but_progressing_stream_is_not_abandoned():
    # The guard must fire on SILENCE, not on slowness: a legitimate NCI1-sized
    # run can take minutes, and killing it would be the opposite of the point.
    def slow():
        for i in range(3):
            time.sleep(0.15)
            yield i

    assert list(_drain(slow(), timeout_s=2)) == [0, 1, 2]


def test_a_silent_stream_is_abandoned_instead_of_waited_on():
    # The failure this exists for: the map produced results for seven minutes,
    # then nothing for three and a half hours while GPU containers stayed
    # allocated and billed.
    def stalls_after_one():
        yield "first"
        time.sleep(30)
        yield "never reached"

    out = []
    with pytest.raises(TimeoutError, match="abandoning this batch"):
        for item in _drain(stalls_after_one(), timeout_s=1):
            out.append(item)
    assert out == ["first"]  # everything already produced is kept


def test_an_exception_from_the_stream_reaches_the_consumer():
    def explodes():
        yield 1
        raise RuntimeError("boom")

    items = list(_drain(explodes(), timeout_s=5))
    assert items[0] == 1
    assert isinstance(items[1], RuntimeError)


def test_an_immediately_empty_stream_terminates():
    assert list(_drain(iter([]), timeout_s=5)) == []
