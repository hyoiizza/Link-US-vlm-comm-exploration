from rover_multi.allocation import merge_candidates, ssi_assign, Task

ROBOTS = ['robot1', 'robot2']


def test_merge_keeps_best_and_records_sources():
    c = [('robot1', 5.0, 0.0, 2.0, '', ''), ('robot2', 5.4, 0.3, 3.0, '', ''),
         ('robot2', 0.0, 8.0, 1.5, '', '')]
    tasks, seq = merge_candidates(c, 1.0)
    assert len(tasks) == 2 and seq == 2
    assert (tasks[0].x, tasks[0].gain, tasks[0].sources) == (5.4, 3.0, ['robot2', 'robot1'])


def _tasks():
    return [Task('t0', 5.4, 0.3, 3.0), Task('t1', 0.0, 8.0, 1.5), Task('t2', -6.0, 0.0, 1.0)]


def test_ssi_gives_contested_task_to_best_bidder():
    u = {('robot1', 't0'): 2.5, ('robot2', 't0'): 2.0, ('robot1', 't1'): 0.5,
         ('robot2', 't1'): 1.2, ('robot1', 't2'): 0.1, ('robot2', 't2'): 0.2}
    a = ssi_assign(ROBOTS, _tasks(), u, 3.0, 0.5)
    assert a['robot1'][0] == 't0' and a['robot2'][0] == 't1'


def test_ssi_spreads_robots_out():
    tasks = _tasks() + [Task('t3', 5.5, 1.0, 2.0)]
    u = {('robot1', 't0'): 2.5, ('robot2', 't0'): 2.0, ('robot2', 't3'): 1.6,
         ('robot1', 't3'): 1.0, ('robot2', 't1'): 1.2}
    assert ssi_assign(ROBOTS, tasks, u, 3.0, 0.5)['robot2'][0] == 't1'


def test_switch_threshold():
    tasks = _tasks() + [Task('t3', 5.5, 1.0, 2.0)]
    busy = {'robot2': ('t1', 1.2)}
    u = {('robot2', 't1'): 1.2, ('robot2', 't3'): 1.3}
    assert ssi_assign(['robot2'], tasks, u, 3.0, 0.5, busy, 0.25)['robot2'][0] == 't1'
    u[('robot2', 't3')] = 1.6
    assert ssi_assign(['robot2'], tasks, u, 3.0, 0.5, busy, 0.25)['robot2'][0] == 't3'


def test_robot_without_bids_gets_nothing():
    a = ssi_assign(ROBOTS, _tasks(), {('robot1', 't0'): 1.0}, 3.0, 0.5)
    assert a == {'robot1': ('t0', 1.0)}
