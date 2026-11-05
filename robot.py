import time
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    # Look up and tilt head
    mini.wake_up()

    # Fire the sound and the motion together — play_sound returns immediately,
    # so the head tilts while the sound plays.
    mini.media.play_sound("hello.wav")
    mini.goto_target(
        head=create_head_pose(z=10, roll=15, degrees=True, mm=True),
        duration=1.0,
    )

    # play_sound is non-blocking: if the clip is longer than the move,
    # pause so it isn't cut off when the process exits.
    time.sleep(1.0)  # ≈ length of hello.wav

    mini.goto_target(head=create_head_pose(), duration=1.0)
    mini.goto_sleep()
