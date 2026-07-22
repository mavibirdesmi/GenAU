import threading
from datetime import datetime, timedelta
from multiprocessing.managers import SyncManager


class YoutubeVideoRateLimitManager:
    """Tracks download start timestamps over a rolling one-hour window and enforces a cap on
    how many downloads may start within any trailing 60 minutes.

    Unlike a fixed-interval limiter (one slot every N seconds, based on an assumed average
    spacing), this keeps the actual timestamps of usages from the last hour, so the enforced
    budget always reflects a real rolling count instead of an approximation.

    An instance of this class is hosted in a multiprocessing manager process (see
    SyncManager.register(...) below); every worker process shares the same proxy, so the
    sliding window is consistent across the whole pool instead of being tracked per worker.
    """

    def __init__(self, limit):
        self.usages = []
        self.limit = limit
        self._lock = threading.Lock()

    def trim_usages(self):
        """Drop usages older than one hour.

        self.usages is kept sorted ascending (oldest first) since record_usage() always
        appends the current time, so we only need to advance past the stale prefix.
        """
        cutoff = datetime.now() - timedelta(hours=1)
        idx = 0
        while idx < len(self.usages) and self.usages[idx] <= cutoff:
            idx += 1
        self.usages = self.usages[idx:]

    def is_under_rate_limit(self):
        self.trim_usages()
        return len(self.usages) < self.limit

    def record_usage(self):
        self.usages.append(datetime.now())

    def try_consume(self):
        """Atomically check-and-record one usage if currently under the limit.

        Combines is_under_rate_limit() + record_usage() under a single lock so concurrent
        callers (each running in a different worker process, but all dispatched onto this
        same manager-process object) cannot race between checking and recording.
        """
        with self._lock:
            if not self.is_under_rate_limit():
                return False
            self.record_usage()
            return True


# Registering on the SyncManager class is what lets ctx.Manager() (used in
# download_audioset_split, in download.py) hand out YoutubeVideoRateLimitManager proxies via
# manager.YoutubeVideoRateLimitManager(...), the same way it already hands out manager.dict()
# and manager.Lock(). The registration must happen at import time, before any manager is
# started, and it re-runs harmlessly in every spawned worker process that imports this module.
SyncManager.register("YoutubeVideoRateLimitManager", YoutubeVideoRateLimitManager)
