"""Use the SO-101 follower's gripper as SteadyHand's hand for a live demo.

The SO-101 has one gripper motor, so grips map to how far it closes:
power closes most, lateral least. The other five joints hold wherever the
arm is. A decision of 0 (hold still) leaves the gripper where it is; "rest"
held for `open_after_s` opens it.

    hand = SO101Hand(port="/dev/ttyACM1", robot_id="kishore_so101")
    hand.connect()
    hand.apply(executed_class)      # call every decision (e.g. every 50 ms)

Uses LeRobot 0.7's SO101Follower (pip install "lerobot[serial-motors]").
Gripper values are in LeRobot's 0-100 range (0 = closed after calibration on
most builds; check yours with lerobot-find-joint-limits and edit CLOSE).
"""
from __future__ import annotations

import time

OPEN = 60.0
CLOSE = {1: 2.0, 2: 25.0, 3: 15.0, 4: 35.0}     # power, pinch, tripod, lateral


class SO101Hand:
    def __init__(self, port: str, robot_id: str, open_after_s: float = 1.0, max_step: float = 8.0):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, max_relative_target=max_step))
        self.open_after_s = open_after_s
        self._rest_since = None
        self._hold = None

    def connect(self):
        self.robot.connect()
        obs = self.robot.get_observation()
        self._hold = {k: v for k, v in obs.items() if k.endswith(".pos")}

    def apply(self, executed: int, user_resting: bool = False):
        now = time.monotonic()
        target = None
        if executed in CLOSE:
            target = CLOSE[executed]
            self._rest_since = None
        elif user_resting:
            self._rest_since = self._rest_since or now
            if now - self._rest_since >= self.open_after_s:
                target = OPEN
        if target is not None:
            self._hold["gripper.pos"] = target
        self.robot.send_action(dict(self._hold))

    def disconnect(self):
        self.robot.disconnect()
