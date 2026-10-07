"""Topic names and QoS of the team (robot <-> robot / coordinator) interface.

Shared by the coordinator (upper layer, leader robot) and the per-robot agents
(lower layer) so both sides always agree. All team topics are absolute
(/team/...), i.e. outside the robot namespaces.

    /team/candidates          TaskCandidates       robot -> coordinator
    /team/auction/announce    AuctionAnnouncement  coordinator -> robots
    /team/auction/bid         Bid                  robot -> coordinator
    /team/auction/award       Award                coordinator -> robots (latched)
    /team/task_status         TaskStatus           robot -> coordinator
    /team/heartbeat           Heartbeat            everyone -> everyone (best effort)
    /team/claim               Claim                robot <-> robot, standalone mode (latched)
    /team/link_quality        LinkQuality          robot -> coordinator / logger (measured Wi-Fi)
"""
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

CANDIDATES = '/team/candidates'
ANNOUNCE = '/team/auction/announce'
BID = '/team/auction/bid'
AWARD = '/team/auction/award'
TASK_STATUS = '/team/task_status'
HEARTBEAT = '/team/heartbeat'
CLAIM = '/team/claim'
LINK_QUALITY = '/team/link_quality'


def _qos(depth, reliable=True, latched=False):
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL if latched else DurabilityPolicy.VOLATILE,
    )


# Must not be lost, but an old one is useless once a newer one exists.
CANDIDATES_QOS = _qos(depth=2)
ANNOUNCE_QOS = _qos(depth=2)
BID_QOS = _qos(depth=10)
TASK_STATUS_QOS = _qos(depth=10)
# Must not be lost, and a robot that (re)connects needs the latest one.
AWARD_QOS = _qos(depth=1, latched=True)
CLAIM_QOS = _qos(depth=1, latched=True)
# Periodic; a dropped one is replaced by the next, so never block Wi-Fi on retransmits.
HEARTBEAT_QOS = _qos(depth=5, reliable=False)
# Periodic measurements; a few lost samples do not matter for the model or the metrics.
LINK_QUALITY_QOS = _qos(depth=20, reliable=False)
