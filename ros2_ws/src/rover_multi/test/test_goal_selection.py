import math

import pytest

from rover_multi.goal_selection import (
    Frontier, joint_assign, merge_reports, normalize, Option, score_options, select_goals, SelectionParams,
    utility)

R = ['robot1', 'robot2']


def test_normalize_clamps_to_unit_range():
    assert normalize(200, 400) == 0.5
    assert normalize(800, 400) == 1.0
    assert normalize(-5, 400) == 0.0


def test_utility_per_method():
    S, G, D = 0.8, 0.4, 0.2
    p = dict(alpha=0.5, lam=0.5)
    # one utility for every method; without semantics S is the constant semantic_default (0.5)
    assert utility(SelectionParams('proposed', **p), S, G, D) == pytest.approx(0.5 * 0.8 + 0.5 * 0.4 - 0.1)
    assert utility(SelectionParams('semantic_only', **p), S, G, D) == pytest.approx(0.5 * 0.8 + 0.5 * 0.4 - 0.1)
    assert utility(SelectionParams('comm_aware', **p), S, G, D) == pytest.approx(0.5 * 0.5 + 0.5 * 0.4 - 0.1)
    assert utility(SelectionParams('frontier', **p), S, G, D) == pytest.approx(0.5 * 0.5 + 0.5 * 0.4 - 0.1)
    for m in ('frontier', 'comm_aware', 'semantic_only', 'proposed'):
        prm = SelectionParams(m, **p)
        assert prm.uses_semantic == (m in ('semantic_only', 'proposed'))
        assert prm.uses_comm_constraint == (m in ('comm_aware', 'proposed'))


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        SelectionParams('greedy')
    with pytest.raises(ValueError):
        SelectionParams(alpha=1.5)


def _scores(u):
    """Hand-made feasible scores {(robot, frontier): utility}."""
    params = SelectionParams('frontier', lam=0.0, g_ref=1.0)
    frontiers = sorted({f for _, f in u})
    fr = [Frontier(f, 0.0) for f in frontiers]
    scores = score_options(params, R, fr, {k: Option(1.0, None) for k in u})
    return {k: s.__class__(**{**s.__dict__, 'utility': u[k]}) if k in u else s for k, s in scores.items()}, frontiers


def test_joint_beats_greedy():
    # Greedy (best single pair first): robot1->a (10), then robot2->b (1) = 11.
    # Joint: robot1->b (9) + robot2->a (9) = 18.
    scores, ids = _scores({('robot1', 'a'): 10, ('robot1', 'b'): 9, ('robot2', 'a'): 9, ('robot2', 'b'): 1})
    assert joint_assign(R, ids, scores) == {'robot1': ('b', 9), 'robot2': ('a', 9)}


def test_distinct_frontiers():
    scores, ids = _scores({('robot1', 'a'): 5, ('robot2', 'a'): 5, ('robot1', 'b'): 0, ('robot2', 'b'): -1})
    out = joint_assign(R, ids, scores)
    assert out['robot1'][0] != out['robot2'][0]


def _frontiers():
    return [Frontier('A', 400, 0.2),   # low relevance, good link
            Frontier('B', 400, 0.9),   # high relevance, good link
            Frontier('C', 800, 0.95)]  # high relevance, bad link, most unknown area


def _options(q_c=-85.0):
    o = {}
    for r in R:
        o[(r, 'A')] = Option(5.0, -55.0)
        o[(r, 'B')] = Option(5.0, -60.0)
        o[(r, 'C')] = Option(5.0, q_c)
    return o


def test_scenario_table():
    """The A/B/C scenario of the paper outline."""
    fr, opt = _frontiers(), _options()
    goals = lambda method, robots=R: {g for g, _ in select_goals(
        SelectionParams(method, alpha=0.5, g_ref=800, q_min=-70), robots, fr, opt)[0].values()}
    assert 'C' in goals('frontier')                   # biggest unknown area, link ignored
    assert goals('comm_aware') == {'A', 'B'}          # C excluded by the link constraint
    assert goals('proposed') == {'A', 'B'}
    # With one robot the semantic term decides between the two reachable frontiers.
    assert goals('proposed', ['robot1']) == {'B'}
    assert goals('comm_aware', ['robot1']) == {'A'}   # tie on G and D -> first frontier


def test_comm_constraint_and_missing_path():
    fr = [Frontier('A', 400, 0.5), Frontier('B', 400, 0.5)]
    opt = {('robot1', 'A'): Option(3.0, -80.0),    # link too weak
           ('robot1', 'B'): Option(None, -50.0),   # no Nav2 path
           ('robot2', 'A'): Option(3.0, None),     # link unknown
           ('robot2', 'B'): Option(3.0, -50.0)}
    out, scores = select_goals(SelectionParams('comm_aware', q_min=-70), R, fr, opt)
    assert scores[('robot1', 'A')].reason == 'link quality below Q_min'
    assert scores[('robot1', 'B')].reason == 'no path'
    assert scores[('robot2', 'A')].reason == 'link quality unknown'
    assert out == {'robot2': ('B', pytest.approx(scores[('robot2', 'B')].utility))}   # robot1 waits
    # The frontier baseline ignores the link but still needs a path.
    out, _ = select_goals(SelectionParams('frontier', q_min=-70), R, fr, opt)
    assert out['robot1'][0] == 'A' and out['robot2'][0] == 'B'


def test_nothing_feasible():
    fr = [Frontier('A', 400, 0.5)]
    opt = {(r, 'A'): Option(3.0, -90.0) for r in R}
    out, _ = select_goals(SelectionParams('proposed', q_min=-70), R, fr, opt)
    assert out == {}


def test_unobserved_frontier_uses_default_and_deterministic_ties():
    fr = [Frontier('A', 400, None), Frontier('B', 400, None)]
    opt = {(r, f): Option(4.0, -50.0) for r in R for f in ('A', 'B')}
    p = SelectionParams('proposed', semantic_default=0.5)
    out1, scores = select_goals(p, R, fr, opt)
    out2, _ = select_goals(p, R, fr, opt)
    assert scores[('robot1', 'A')].S == 0.5
    assert out1 == out2 and out1['robot1'][0] == 'A'
    assert not math.isinf(out1['robot1'][1])


def test_merge_reports():
    from rover_multi.goal_selection import merge_reports
    f = merge_reports('t1', [('robot1', {'unknown_cells': 300, 'semantic': 0.8}),
                             ('robot2', {'unknown_cells': 120, 'semantic': None}),
                             ('robot2', {'unknown_cells': 200, 'semantic': 0.6})])
    assert f.unknown_cells == 120 and abs(f.semantic - 0.7) < 1e-9
    assert merge_reports('t2', [('robot1', {'unknown_cells': 50, 'semantic': None})]).semantic is None


def test_merge_candidates_keeps_reports():
    from rover_multi.allocation import merge_candidates
    tasks, _ = merge_candidates([('robot1', 0.0, 0.0, 2.0, '', '', {'unknown_cells': 10}),
                                 ('robot2', 0.3, 0.0, 1.0, '', '', {'unknown_cells': 5})], 1.0)
    assert len(tasks) == 1 and [r for r, _ in tasks[0].reports] == ['robot1', 'robot2']


def test_keep_bonus_holds_current_goal():
    params = SelectionParams(method='frontier', lam=0.0, g_ref=100.0)
    frontiers = [Frontier('a', 50.0), Frontier('b', 55.0)]
    options = {('r1', 'a'): Option(5.0, -50.0), ('r1', 'b'): Option(5.0, -50.0)}
    chosen, _ = select_goals(params, ['r1'], frontiers, options)
    assert chosen['r1'][0] == 'b'
    chosen, _ = select_goals(params, ['r1'], frontiers, options, current={'r1': 'a'}, keep_bonus=0.1)
    assert chosen['r1'][0] == 'a'
    assert abs(chosen['r1'][1] - 0.5) < 1e-9     # reported utility has no bonus in it


def test_unobserved_frontier_gets_mean_view_prior():
    params = SelectionParams(method='proposed', alpha=0.5, lam=0.0, g_ref=100.0)
    seen = merge_reports('a', [('r1', {'unknown_cells': 50.0, 'semantic': 0.46})])
    unseen = merge_reports('b', [('r1', {'unknown_cells': 50.0, 'semantic': None, 'semantic_prior': 0.44})])
    options = {('r1', 'a'): Option(5.0, -50.0), ('r1', 'b'): Option(5.0, -50.0)}
    chosen, scores = select_goals(params, ['r1'], [seen, unseen], options)
    assert abs(scores[('r1', 'b')].S - 0.44) < 1e-9
    assert chosen['r1'][0] == 'a'            # seen and above the average view wins
    const = SelectionParams(method='proposed', alpha=0.5, lam=0.0, g_ref=100.0, unobserved_semantic='constant')
    chosen, scores = select_goals(const, ['r1'], [seen, unseen], options)
    assert scores[('r1', 'b')].S == 0.5 and chosen['r1'][0] == 'b'


def test_semantic_only_uses_s_but_ignores_link():
    fr = [Frontier('A', 400, 0.2), Frontier('B', 400, 0.9)]
    opt = {('robot1', 'A'): Option(5.0, -50.0), ('robot1', 'B'): Option(5.0, -90.0)}   # B: weak link
    pick = lambda m: select_goals(SelectionParams(m, q_min=-70), ['robot1'], fr, opt)[0]['robot1'][0]
    assert pick('semantic_only') == 'B'      # higher S, link ignored
    assert pick('proposed') == 'A'           # B excluded by Q_min
    assert pick('frontier') == 'A'           # S constant: tie on G and D -> first frontier
