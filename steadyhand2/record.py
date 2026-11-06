"""Record EMG and wrist video together, with on-screen cues.

    python -m steadyhand2.record --device oymotion --camera 0 --out data/s01 --reps 5

For each object card in the protocol the screen says what to do ("REST",
then "GRAB the mug: power grip", ...). Every EMG chunk and every frame is
stored with a time.monotonic() timestamp from the same clock, plus the cue
active at that moment, so EMG windows and frames can be matched later.

EMG comes from a LibEMG streamer (2.0.3 supports SiFi BioArmband/BioPoint,
Myo, Delsys, OyMotion gForce, EMaGer, MindRove, OTB Muovi+). LibEMG keeps
the NEWEST samples at the TOP of its shared buffer; `drain()` handles that
and the sample counter so nothing is read twice or skipped.

Recording other people needs their written consent and, for anything you
publish, usually an ethics review (IRB). Start with yourself.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PROTOCOL = [  # (cue text, label: 0 = rest, else grip index into vision.GRIPS + 1)
    ("REST - relax your hand", 0),
    ("GRAB the water bottle (power)", 1),
    ("REST", 0),
    ("GRAB the coin / pen cap (pinch)", 2),
    ("REST", 0),
    ("GRAB the marker (tripod)", 3),
    ("REST", 0),
    ("GRAB the key / card (lateral)", 4),
]


def make_streamer(device: str):
    from libemg import streamers
    fn = {"oymotion": streamers.oymotion_streamer, "myo": streamers.myo_streamer,
          "mindrove": streamers.mindrove_streamer, "sifi": streamers.sifi_bioarmband_streamer,
          "delsys": streamers.delsys_streamer, "emager": streamers.emager_streamer}[device]
    return fn()


def new_rows(buf: np.ndarray, prev_top: np.ndarray | None, count_diff: int) -> int:
    """How many rows at the top of a newest-first buffer are new.

    LibEMG streamers disagree on what `emg_count` counts (MindRove adds 1 per
    write, Myo 2 per packet, SiFi the number of samples), so the count alone
    can drop or repeat samples. Instead, find where the rows we already read
    (`prev_top`, the previous newest rows) now sit in the buffer; everything
    above them is new. Fall back to the count only if they can't be found
    (first read, or the buffer overflowed between reads)."""
    if prev_top is None or len(prev_top) == 0:
        return min(max(count_diff, 0), len(buf))
    k = len(prev_top)
    for i in range(0, len(buf) - k + 1):
        if np.array_equal(buf[i:i + k], prev_top):
            return i
    return min(max(count_diff, 0), len(buf))


class EMGReader:
    def __init__(self, device: str, match_rows: int = 8):
        from libemg.data_handler import OnlineDataHandler
        self.proc, smi = make_streamer(device)
        self.odh = OnlineDataHandler(smi)
        self.last_count = 0
        self.prev_top = None
        self.match_rows = match_rows

    def drain(self):
        """New samples since the last call, oldest first: (n, channels)."""
        data, count = self.odh.get_data(filter=False)
        emg, c = np.asarray(data["emg"]), int(np.ravel(count["emg"])[0])
        n = new_rows(emg, self.prev_top, c - self.last_count)
        self.last_count = c
        if n > 0 or self.prev_top is None:
            self.prev_top = emg[: self.match_rows].copy()
        return emg[:n][::-1] if n > 0 else np.zeros((0, emg.shape[1]))


def main(argv=None):
    import cv2

    p = argparse.ArgumentParser()
    p.add_argument("--device", default="oymotion")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--cue_s", type=float, default=4.0)
    p.add_argument("--fps", type=float, default=15.0)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    emg = EMGReader(a.device)
    cam = cv2.VideoCapture(a.camera)
    emg_chunks, emg_t, emg_lab, frames, frame_t, frame_lab = [], [], [], [], [], []
    try:
        for rep in range(a.reps):
            for text, lab in PROTOCOL:
                t_end = time.monotonic() + a.cue_s
                next_frame = 0.0
                while (now := time.monotonic()) < t_end:
                    chunk = emg.drain()
                    if len(chunk):
                        emg_chunks.append(chunk)
                        emg_t.append(np.full(len(chunk), now))
                        emg_lab.append(np.full(len(chunk), lab))
                    if now >= next_frame:
                        ok, fr = cam.read()
                        if ok:
                            frames.append(cv2.resize(fr, (320, 240))[:, :, ::-1])   # RGB
                            frame_t.append(now)
                            frame_lab.append(lab)
                            show = fr.copy()
                            cv2.putText(show, f"{rep + 1}/{a.reps}  {text}", (10, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255) if lab else (0, 160, 0), 2)
                            cv2.imshow("steadyhand2 recorder", show)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                raise KeyboardInterrupt
                        next_frame = now + 1.0 / a.fps
                    time.sleep(0.002)
    except KeyboardInterrupt:
        print("stopped early")
    finally:
        cam.release()
        cv2.destroyAllWindows()
    np.savez_compressed(out / "session.npz",
                        emg=np.concatenate(emg_chunks), emg_t=np.concatenate(emg_t), emg_label=np.concatenate(emg_lab),
                        frames=np.stack(frames), frame_t=np.array(frame_t), frame_label=np.array(frame_lab))
    (out / "info.json").write_text(json.dumps({"device": a.device, "protocol": PROTOCOL, "reps": a.reps,
                                               "cue_s": a.cue_s, "fps": a.fps}, indent=2))
    print("saved", out / "session.npz")


if __name__ == "__main__":
    main()
