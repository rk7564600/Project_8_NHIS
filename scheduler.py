"""Monthly refresh loop for the `scheduler` container: re-runs the pipeline on the 1st of each month
(and once at start-up if no output exists). Uses only the standard library."""
import os
import time
from datetime import datetime

from . import pipeline, prep

CHECK_EVERY = int(os.environ.get("CHECK_EVERY_SECONDS", 3600))


def main():
    last = None
    if not (prep.OUTPUT_DIR / "metrics.json").exists():
        pipeline.run(); last = datetime.now().strftime("%Y-%m")
    while True:
        now = datetime.now()
        key = now.strftime("%Y-%m")
        if now.day == 1 and key != last:
            print(f"monthly refresh {key}", flush=True)
            try:
                pipeline.run(); last = key
            except Exception as e:           # keep the loop alive; the previous outputs stay in place
                print("refresh failed:", e, flush=True)
        time.sleep(CHECK_EVERY)


if __name__ == "__main__":
    main()
