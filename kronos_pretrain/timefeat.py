"""Time features (train.time_features).
kronos  : minute, hour, weekday, day, month  (upstream)
session : minute -> floor(59 * elapsed / session_length) within the bar's trading session, hour -> session index;
          weekday/day/month unchanged. Gives 24h markets and multi-session markets one shared semantics.
          Sessions per market in train.sessions, e.g. CN: [["09:30","11:30"],["13:00","15:00"]]. Markets without an
          entry keep the kronos columns."""
import numpy as np

def _mins(s): h, m = s.split(":"); return int(h) * 60 + int(m)

def session_columns(ts_sec, sessions):
    """ts_sec int64 local wall clock -> (minute_col, hour_col) int16. Bars are labelled at bar END."""
    tod = (ts_sec % 86400) // 60                                  # minute of day
    segs = [(_mins(a), _mins(b)) for a, b in sessions]
    total = sum(b - a for a, b in segs)
    elapsed = np.zeros_like(tod); sidx = np.zeros_like(tod); acc = 0
    for i, (a, b) in enumerate(segs):
        m = (tod > a) & (tod <= b)
        elapsed[m] = acc + (tod[m] - a); sidx[m] = i; acc += b - a
    after_last = tod > segs[-1][1]; elapsed[after_last] = total; sidx[after_last] = len(segs) - 1
    minute = np.clip(np.floor(59.0 * elapsed / max(total, 1)), 0, 59).astype(np.int16)
    return minute, np.clip(sidx, 0, 23).astype(np.int16)
