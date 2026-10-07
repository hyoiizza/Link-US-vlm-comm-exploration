"""Joint frontier goal selection for the team, free of ROS so it can be unit tested.

Scores every (robot i, frontier j) pair and picks distinct goals for the robots:

    U_ij = alpha * S_j + (1 - alpha) * G_j - lambda * D_ij        (every method)
    subject to  path(i, j) exists  [and  Q_ij >= Q_min]

    S_j   semantic relevance of frontier j for the search task [0-1] (VLM); a frontier
          without an observation gets the robots' mean S over all their views (semantic
          prior, unobserved_semantic 'mean_view') or the constant `semantic_default`
    G_j   = min(N_unknown(j) / G_ref, 1)    normalized exploration gain
    D_ij  = min(path_cost(i, j) / D_ref, 1)  normalized travel cost (path length + turn cost)
    Q_ij  predicted link quality at the goal (larger is better, e.g. RSSI dBm)

Communication is a feasibility constraint, not a reward term. All methods use the
same utility, weights and assignment; they differ only in whether S is used (methods
without it get the constant S = semantic_default for every frontier, which cannot change
a choice) and whether Q is enforced, so S and Q effects can be separated:

    method          S used   Q >= Q_min
    frontier          no        no
    comm_aware        no        yes
    semantic_only     yes       no
    proposed          yes       yes

(Before 2026-10-01 frontier/comm_aware used G - lambda*D and the ablation
proposed_no_semantic (1-alpha)*G - lambda*D, i.e. a G:D weighting that differed by
1/(1-alpha) from proposed, which confounded the S effect with the distance weight.)

Assignment maximizes the summed utility over robots with distinct frontiers
(j != k for two robots). If not every robot can get a feasible frontier, the
largest number of robots is served and the others are left without a goal
(the caller sends them to wait where the link is good).
"""
from dataclasses import dataclass
from itertools import combinations, permutations
import math

METHODS = ('frontier', 'comm_aware', 'semantic_only', 'proposed')


@dataclass(frozen=True)
class SelectionParams:
    method: str = 'proposed'
    alpha: float = 0.5            # semantic relevance vs exploration gain
    lam: float = 1.0              # travel cost weight (lambda)
    g_ref: float = 700.0          # unknown cells counted as full gain (G = 1)
    d_ref: float = 25.0           # path cost [m] counted as full cost (D = 1)
    q_min: float = -70.0          # minimum predicted link quality (e.g. RSSI dBm)
    semantic_default: float = 0.5  # S of a frontier that has no observation yet (and no prior)
    unobserved_semantic: str = 'mean_view'  # 'mean_view': the robots' mean S; 'constant': semantic_default

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f'unknown selection method {self.method!r}, expected one of {METHODS}')
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError('alpha must be in [0, 1]')
        if self.unobserved_semantic not in ('mean_view', 'constant'):
            raise ValueError("unobserved_semantic must be 'mean_view' or 'constant'")
        if self.lam < 0.0 or self.g_ref <= 0.0 or self.d_ref <= 0.0:
            raise ValueError('lam must be >= 0 and g_ref, d_ref > 0')

    @property
    def uses_comm_constraint(self):
        return self.method in ('comm_aware', 'proposed')

    @property
    def uses_semantic(self):
        return self.method in ('semantic_only', 'proposed')


@dataclass(frozen=True)
class Frontier:
    id: str
    unknown_cells: float          # raw gain N_unknown
    semantic: float = None        # S_j in [0, 1], None when not observed
    semantic_prior: float = None  # mean S of the reporting robots' views, None when unknown


@dataclass(frozen=True)
class Option:
    """What one robot reported about one frontier."""
    path_length: float = None     # [m], None when Nav2 found no path
    link_quality: float = None    # predicted Q_ij, None when unknown


@dataclass(frozen=True)
class Score:
    """Everything that went into one (robot, frontier) decision, for logging."""
    robot: str
    frontier: str
    S: float
    G: float
    D: float
    Q: float
    feasible: bool
    reason: str                   # '' when feasible, else why it was excluded
    utility: float


def merge_reports(frontier_id, reports):
    """One Frontier from the reports of the robots that saw it: [(robot, dict)].

    dict keys: unknown_cells, semantic (None when not observed), semantic_prior (optional).
    N_unknown is the minimum over the robots (a cell one robot has mapped is not unknown),
    S is the mean over the robots that observed the frontier, the prior the mean over the others.
    """
    unknown = min((r['unknown_cells'] for _, r in reports), default=0.0)
    observed = [r['semantic'] for _, r in reports if r.get('semantic') is not None]
    semantic = sum(observed) / len(observed) if observed else None
    priors = [r['semantic_prior'] for _, r in reports if r.get('semantic_prior') is not None]
    prior = sum(priors) / len(priors) if priors else None
    return Frontier(frontier_id, unknown, semantic, prior)


def normalize(value, ref):
    return min(max(value, 0.0) / ref, 1.0)


def utility(params, S, G, D):
    """Same for every method; methods without semantics see the constant S = semantic_default."""
    if not params.uses_semantic:
        S = params.semantic_default
    return params.alpha * S + (1.0 - params.alpha) * G - params.lam * D


def score_options(params, robots, frontiers, options):
    """Scores for every robot x frontier.

    options: {(robot, frontier_id): Option}; a missing entry means the robot did not
    evaluate that frontier and it is infeasible for it.
    Returns {(robot, frontier_id): Score}.
    """
    scores = {}
    for f in frontiers:
        if not params.uses_semantic:
            S = params.semantic_default
        elif f.semantic is not None:
            S = min(max(f.semantic, 0.0), 1.0)
        elif params.unobserved_semantic == 'mean_view' and f.semantic_prior is not None:
            S = min(max(f.semantic_prior, 0.0), 1.0)
        else:
            S = params.semantic_default
        G = normalize(f.unknown_cells, params.g_ref)
        for robot in robots:
            opt = options.get((robot, f.id))
            reason = ''
            D = Q = math.nan
            if opt is None:
                reason = 'not evaluated'
            elif opt.path_length is None:
                reason = 'no path'
            else:
                D = normalize(opt.path_length, params.d_ref)
                Q = math.nan if opt.link_quality is None else opt.link_quality
                if params.uses_comm_constraint and not (opt.link_quality is not None
                                                        and opt.link_quality >= params.q_min):
                    reason = 'link quality below Q_min' if opt.link_quality is not None else 'link quality unknown'
            feasible = reason == ''
            U = utility(params, S, G, D) if feasible else -math.inf
            scores[(robot, f.id)] = Score(robot, f.id, S, G, D, Q, feasible, reason, U)
    return scores


def joint_assign(robots, frontier_ids, scores, bonus=None):
    """Distinct feasible frontiers for as many robots as possible, maximizing the summed utility.

    For two robots this is  argmax_{j != k} U_1j + U_2k  over feasible pairs, falling back
    to the best single robot-frontier pair when no feasible pair exists. Exhaustive, so it
    is meant for a handful of robots and tens of frontiers. Ties are broken by robot and
    frontier order so the result is reproducible.

    bonus: {(robot, frontier_id): b} added to U only while comparing assignments
    (e.g. keep a busy robot on its current goal unless another beats it by b).

    Returns {robot: (frontier_id, utility)}; robots without a goal are absent.
    """
    bonus = bonus or {}
    robots = list(robots)
    frontier_ids = list(frontier_ids)
    usable = {r: [f for f in frontier_ids if scores.get((r, f)) is not None and scores[(r, f)].feasible]
              for r in robots}
    for n in range(len(robots), 0, -1):
        best = None
        for group in combinations(robots, n):
            if any(not usable[r] for r in group):
                continue
            for goals in permutations(frontier_ids, n):
                if any(g not in usable[r] for r, g in zip(group, goals)):
                    continue
                total = sum(scores[(r, g)].utility + bonus.get((r, g), 0.0) for r, g in zip(group, goals))
                if best is None or total > best[0] + 1e-12:
                    best = (total, group, goals)
        if best is not None:
            _, group, goals = best
            return {r: (g, scores[(r, g)].utility) for r, g in zip(group, goals)}
    return {}


def select_goals(params, robots, frontiers, options, current=None, keep_bonus=0.0):
    """score_options + joint_assign. Returns (assignment, scores).

    current: {robot: frontier_id} goals the robots are driving to; they get keep_bonus
    in the assignment so a re-selection only switches for a clearly better goal."""
    scores = score_options(params, robots, frontiers, options)
    bonus = {(r, f): keep_bonus for r, f in (current or {}).items()}
    return joint_assign(robots, [f.id for f in frontiers], scores, bonus), scores
