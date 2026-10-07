"""Rewrite single-robot parameter files for a namespaced robot.

All robots share the global /tf tree, so every frame gets a "<robot>/" prefix,
and fully qualified topics ("/scan") are moved under the robot namespace
("/robot1/scan"). Relative topics already resolve inside the namespace.
"""
import re
import tempfile

import yaml

# Parameter names that hold a TF frame id (base_frame_id, odom_frame, map_frame, global_frame, ...).
_FRAME_KEY = re.compile(r'(^|_)frame(_id)?$')
# Parameter names that hold a topic name (scan_topic, odom_topic, topic, map_name, ...).
_TOPIC_KEY = re.compile(r'(^|_)topic$|^map_name$')


def prefix_frame(robot, frame):
    if not frame or frame.startswith(robot + '/'):
        return frame
    return f'{robot}/{frame.lstrip("/")}'


def namespace_topic(robot, topic):
    if topic.startswith('/') and not topic.startswith(f'/{robot}/'):
        return f'/{robot}{topic}'
    return topic


def _rewrite(node, robot):
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            skey = str(key)
            if skey.startswith('qos_overrides./'):
                skey = 'qos_overrides.' + namespace_topic(robot, skey[len('qos_overrides.'):])
            if isinstance(value, str) and _FRAME_KEY.search(skey):
                value = prefix_frame(robot, value)
            elif isinstance(value, str) and _TOPIC_KEY.search(skey):
                value = namespace_topic(robot, value)
            else:
                value = _rewrite(value, robot)
            out[skey] = value
        return out
    return node


def _merge(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _is_ros_params_file(data):
    if not isinstance(data, dict):
        return False
    return 'ros__parameters' in data or any(_is_ros_params_file(v) for v in data.values())


def rewrite_params_file(path, robot, overrides=None, wrap_namespace=True):
    """Return the path of a temporary copy of ``path`` rewritten for ``robot``.

    ``overrides`` is merged after rewriting, e.g.
    ``{'slam_toolbox': {'ros__parameters': {'map_name': '/robot1/map'}}}``.

    A ROS parameter file keyed by bare node names ("ekf_filter_node:") only
    matches nodes in the root namespace, so its content is nested under the
    robot namespace ("robot1: ekf_filter_node:"). Pass ``wrap_namespace=False``
    when the consumer does that itself (Nav2's RewrittenYaml root_key).
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    data = _rewrite(data, robot)
    if overrides:
        data = _merge(data, overrides)
    if wrap_namespace and _is_ros_params_file(data):
        data = {robot: data}
    tmp = tempfile.NamedTemporaryFile(
        mode='w', prefix=f'{robot}_', suffix='_' + path.rsplit('/', 1)[-1], delete=False)
    yaml.safe_dump(data, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()
    return tmp.name
