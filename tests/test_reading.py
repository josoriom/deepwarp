import threading
import time

import numpy as np
import pytest

import deepwarp

options = {"points": 8}


def read_file(number, options):
    time.sleep(0.01)
    return np.full((3, options["points"]), number, dtype=np.float32), np.full(3, number, dtype=np.float32)


def test_gives_groups_in_file_order():
    groups = list(deepwarp.read(read_file, range(10), options, group_size=4, readers=3))
    assert [len(dataset) for dataset, _ in groups] == [4, 4, 2]
    assert [dataset[:, 0, 0].tolist() for dataset, _ in groups] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert groups[0][1].shape == (4, 3)


def test_reads_the_next_group_while_the_loop_works():
    started = []

    def read_and_note(number, options):
        started.append(number)
        return read_file(number, options)

    reading = deepwarp.read(read_and_note, range(8), options, group_size=2, readers=2)
    next(reading)
    time.sleep(0.2)
    assert len(started) > 2
    reading.close()


def test_a_failing_file_stops_the_loop_and_is_named():
    def read_or_fail(number, options):
        if number == 5:
            raise OSError("broken file")
        return read_file(number, options)

    with pytest.raises(deepwarp.ReadError, match="could not read 5"):
        for _ in deepwarp.read(read_or_fail, range(10), options, group_size=2, readers=2):
            pass


def test_leaving_the_loop_early_stops_the_readers():
    before = threading.active_count()
    for _ in deepwarp.read(read_file, range(100), options, group_size=2, readers=4):
        break
    time.sleep(0.1)
    assert threading.active_count() <= before


def test_every_file_gets_the_options():
    groups = list(deepwarp.read(read_file, range(3), {"points": 5}, group_size=3, readers=2))
    assert groups[0][0].shape == (3, 3, 5)
