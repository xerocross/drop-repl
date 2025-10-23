#!/usr/bin/env python3

from __future__ import annotations
import shlex
from datetime import timedelta, date


import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

# =========================
# CONFIG
# =========================
# Set this to wherever you want your data to live:
odin_root = Path("~/Documents/Odin").expanduser()

STORAGE_PATH = odin_root / "drops.jsonl"
EXPORT_DIR = odin_root / "Notes" / "Drops"


# =========================
# =========================

HASHTAG_RE = re.compile(r"#([\w\-]+)")

@dataclass
class Scrap:
    id: int
    ts: str          # ISO 8601 UTC string
    text: str
    tags: List[str]

    @staticmethod
    def from_line(line: str) -> "Scrap":
        obj = json.loads(line)
        return Scrap(
            id=int(obj["id"]),
            ts=str(obj["ts"]),
            text=str(obj["text"]),
            tags=list(obj.get("tags", [])),
        )

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False) + "\n"


# =========================
# STORAGE
# =========================

def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

class ScrapStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

        self._deleted = set()   # IDs that are tombstoned
        self._next_id = 1
        self._bootstrap_from_disk()

        # append handle
        self._fh = open(self.path, "a", encoding="utf-8")

    def _bootstrap_from_disk(self) -> None:
        """Scan once: discover deleted IDs and compute next_id."""
        max_id = 0
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue

                    # Tombstone?
                    if isinstance(obj, dict) and obj.get("op") == "del":
                        try:
                            self._deleted.add(int(obj["id"]))
                        except Exception:
                            pass
                        continue

                    # Regular scrap line
                    try:
                        s = Scrap(
                            id=int(obj["id"]),
                            ts=str(obj["ts"]),
                            text=str(obj["text"]),
                            tags=list(obj.get("tags", [])),
                        )
                        if s.id > max_id:
                            max_id = s.id
                    except Exception:
                        continue
        except FileNotFoundError:
            pass

        self._next_id = max_id + 1

    def _scan_next_id(self) -> int:
        """Find the next ID by scanning the file once at startup."""
        max_id = 0
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        s = Scrap.from_line(line)
                        if s.id > max_id:
                            max_id = s.id
                    except Exception:
                        # Skip malformed lines rather than crashing
                        continue
        except FileNotFoundError:
            pass
        return max_id + 1

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def add(self, text: str) -> Scrap:
        tags = sorted(set([m.group(1).lower() for m in HASHTAG_RE.finditer(text)]))
        scrap = Scrap(
            id=self._next_id,
            ts=_utc_now_iso(),
            text=text,
            tags=tags,
        )
        self._next_id += 1
        self._write_line(scrap.to_jsonl())
        return scrap
    
    def delete(self, scrap_id: int) -> bool:
        """Append a tombstone for the given id if it exists and isn't already deleted."""
        if scrap_id in self._deleted:
            return False  # already deleted
        exists_live = any(s.id == scrap_id for s in self.iter_live())
        if not exists_live:
            return False
        self._write_tombstone(scrap_id)
        self._deleted.add(scrap_id)
        return True

    def _write_line(self, line: str) -> None:
        self._fh.write(line)
        self._fh.flush()
        # ensure durability; cost is minimal for a quick log
        os.fsync(self._fh.fileno())

    def iter_all(self) -> Iterable[Scrap]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Scrap.from_line(line)
                except Exception:
                    continue

    # Queries:

    def search_substring(self, needle: str) -> Iterable[Scrap]:
        n = needle.lower()
        for s in self.iter_live():
            if n in s.text.lower():
                yield s

    def search_tag(self, tag: str) -> Iterable[Scrap]:
        t = tag.lower().lstrip("#")
        for s in self.iter_live():
            if t in s.tags:
                yield s

    def last_n(self, n: int = 10) -> List[Scrap]:
        items = list(self.iter_live())
        return items[-n:]

    def _write_line(self, line: str) -> None:
        self._fh.write(line)
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def _write_tombstone(self, scrap_id: int) -> None:
        tomb = {"op": "del", "id": scrap_id, "ts": _utc_now_iso()}
        self._write_line(json.dumps(tomb, ensure_ascii=False) + "\n")

    def iter_all(self) -> Iterable[Scrap]:
        """All scraps ever written (including ones later deleted)."""
        with open(self.path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("op") == "del":
                    # skip tombstones in this iterator
                    continue
                try:
                    yield Scrap.from_line(json.dumps(obj))
                except Exception:
                    continue

    def iter_live(self) -> Iterable[Scrap]:
        """Only scraps that are NOT deleted."""
        deleted = self._deleted
        for s in self.iter_all():
            if s.id not in deleted:
                yield s



# =========================
# REPL
# =========================

HELP_TEXT = """Commands:
  h                      : help
  r <text>               : record a scrap (everything after 'r' is the note). #hashtags are auto-detected.
  s <substring>          : search by substring (case-insensitive)
  t <tag>                : search by tag (with or without leading #)
  ls [n]                  : list last n scraps (default 10)
  f [filters]            : interactive filter with state (intersect on each call)
                           -t <tag>  (repeatable; with or without #)
                           -s <text> (substring, caseel-insensitive)
                           --all     (seed state to all live scraps)
                           --reset   (clear state)
  fs                     : show current filter state (same as 'f' with no args)
  c                      : clear filter state (same as 'f --reset')
  d <id>                 : delete a scrap by id
  x                      : export today's drops to a markdown file
  x -y                   : export yesterday's drops
  q                      : quit

Examples:
  r Met a lovely barista #people #coffee
  s barista
  t #coffee
  l 25
"""

def print_scrap(s: Scrap) -> None:
    ts_local = _to_local(s.ts)
    tag_str = (" #" + " #".join(s.tags)) if s.tags else ""
    print(f"[{s.id}] {ts_local}{tag_str}\n    {s.text}")

def _to_local(ts_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return ts_iso


class SessionState:
    """Holds the current filtered set of scraps for interactive narrowing."""
    def __init__(self):
        self.current: Optional[List[Scrap]] = None

    def seed_all(self, store: "ScrapStore"):
        self.current = list(store.iter_live())

    def reset(self):
        self.current = None

    def base_candidates(self, store: "ScrapStore") -> List[Scrap]:
        return self.current if self.current is not None else list(store.iter_live())

def _normalize_tag(t: str) -> str:
    return t.lstrip("#").lower()

def _matches(scrap: Scrap, tags: List[str], substr: Optional[str]) -> bool:
    if tags:
        stags = set(scrap.tags)
        # Require ALL tags to be present
        for t in tags:
            if t not in stags:
                return False
    if substr:
        if substr.lower() not in scrap.text.lower():
            return False
    return True

def _apply_filter(candidates: List[Scrap], tags: List[str], substr: Optional[str]) -> List[Scrap]:
    return [s for s in candidates if _matches(s, tags, substr)]

def _local_date_str(ts_iso: str) -> str:
    """YYYY-MM-DD in *local* time for the given ISO timestamp."""
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d")
    except Exception:
        # Fallback: trust the leading date part if parse fails
        return ts_iso[:10]

def _local_time_str(ts_iso: str) -> str:
    """HH:MM in local time for pretty printing."""
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%H:%M")
    except Exception:
        return "??:??"

def export_daily_markdown(store: "ScrapStore", export_dir: Path, days_ago: int = 0) -> Path:
    """
    Export all *live* scraps from a given local calendar day to Markdown.
    days_ago=0 => today, 1 => yesterday, etc.
    """
    target_day = (date.today() - timedelta(days=days_ago)).isoformat()
    export_dir = export_dir.expanduser().resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    # Collect scraps for that local day
    daily = []
    for s in store.iter_live():
        if _local_date_str(s.ts) == target_day:
            daily.append(s)
    # Sort by timestamp just in case
    daily.sort(key=lambda s: s.ts)

    # Build markdown
    lines = [f"# Drops from {target_day}", ""]
    for s in daily:
        t = _local_time_str(s.ts)
        lines.append(f"- [{t}] {s.text}")
    lines.append("")  # trailing newline

    out_path = export_dir / f"drop-{target_day}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path

def repl(store: ScrapStore) -> None:
    state = SessionState()
    print(f"Scraps REPL. Storing to: {store.path}")
    print("Type 'h' for help.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        # Split into command and remainder
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        # Allow forms like d6, l25, etc.
        if len(cmd) > 1 and cmd[0] in ("d", "l", "r", "s", "t", "f"):
            # peel off the first char as command, rest as argument
            arg = cmd[1:] + (" " + arg if arg else "")
            cmd = cmd[0]

        if cmd in ("h", "help", "?"):
            print(HELP_TEXT)
        elif cmd in ("c", "clear"):
            state.reset()
            print("Filter state cleared.")
        elif cmd in ("x",):
            # simple: 'x' for today, 'x -y' for yesterday
            yesterday = "-y" in arg.split() if arg else False
            days_ago = 1 if yesterday else 0
            out_path = export_daily_markdown(store, EXPORT_DIR, days_ago=days_ago)
            # Count how many actually exported so we can print a friendly summary
            target_day = (date.today() - timedelta(days=days_ago)).isoformat()
            count = sum(1 for s in store.iter_live() if _local_date_str(s.ts) == target_day)
            print(f"Exported {count} drops to {out_path}")
        elif cmd in ("fs",):
            # show current state
            items = state.base_candidates(store) if state.current is not None else []
            if not items:
                print("(state is empty)" if state.current is not None else "(no state; use 'f --all' or run a filter)")
            else:
                for s in items:
                    print_scrap(s)
        elif cmd in ("f", "filter"):
            if not arg:
                # show state
                items = state.base_candidates(store) if state.current is not None else []
                if not items:
                    print("(state is empty)" if state.current is not None else "(no state; use 'f --all' or add filters)")
                else:
                    for s in items:
                        print_scrap(s)
                continue

            # parse args with shlex to allow quotes in substring
            try:
                tokens = shlex.split(arg)
            except ValueError as e:
                print(f"Parse error: {e}")
                continue

            # flags
            if "--reset" in tokens:
                state.reset()
                print("Filter state cleared.")
                continue
            if "--all" in tokens:
                state.seed_all(store)
                print("Filter state seeded to all live scraps.")
                continue

            # collect filters
            tags: List[str] = []
            substr: Optional[str] = None

            i = 0
            while i < len(tokens):
                t = tokens[i]
                if t == "-t":
                    if i + 1 >= len(tokens):
                        print("Usage: f -t <tag> [...]")
                        break
                    tags.append(_normalize_tag(tokens[i+1]))
                    i += 2
                elif t == "-s":
                    if i + 1 >= len(tokens):
                        print("Usage: f -s <substring>")
                        break
                    substr = tokens[i+1]
                    i += 2
                else:
                    print(f"Unknown flag '{t}'. Use -t, -s, --all, or --reset.")
                    break
            else:
                # only runs if 'break' never hit
                base = state.base_candidates(store)
                result = _apply_filter(base, tags, substr)
                state.current = result
                if not result:
                    print("No matches. (state now empty)")
                else:
                    for s in result:
                        print_scrap(s)
                continue
            # if we got here, there was a usage error—don’t change state
        elif cmd in ("q", "quit", "exit"):
            break
        elif cmd in ("r", "rec", "record"):
            if not arg:
                print("Nothing to record. Usage: r <text>")
                continue
            scrap = store.add(arg)
            print("Recorded:")
            print_scrap(scrap)
        elif cmd in ("s", "search"):
            if not arg:
                print("Usage: s <substring>")
                continue
            hits = list(store.search_substring(arg))
            if not hits:
                print("No matches.")
            else:
                for s in hits:
                    print_scrap(s)
        elif cmd in ("t", "tag"):
            if not arg:
                print("Usage: t <tag>")
                continue
            hits = list(store.search_tag(arg))
            if not hits:
                print("No matches.")
            else:
                for s in hits:
                    print_scrap(s)
        elif cmd in ("d", "del", "delete"):
            if not arg:
                print("Usage: d <id>")
                continue
            try:
                sid = int(arg)
            except ValueError:
                print("ID must be an integer.")
                continue
            ok = store.delete(sid)
            if ok:
                print(f"Deleted scrap {sid}.")
            else:
                print(f"No live scrap with id {sid} (maybe already deleted?).")

        elif cmd in ("l", "ls", "list"):
            n = 10
            if arg:
                try:
                    n = max(1, int(arg))
                except ValueError:
                    print("Provide a number, e.g., 'l 25'")
                    continue
            items = store.last_n(n)
            if not items:
                print("No scraps yet.")
            else:
                for s in items:
                    print_scrap(s)
        else:
            print("Unknown command. Type 'h' for help.")

def main() -> None:
    store = ScrapStore(STORAGE_PATH)
    try:
        repl(store)
    finally:
        store.close()

if __name__ == "__main__":
    main()

