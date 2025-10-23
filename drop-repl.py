#!/usr/bin/env python3
# drop-repl.py — “Drop” (drp/drop)
# Default: one-shot capture; subcommands for everything else.
# Python 3.9+. No hard deps; optional: rich, rapidfuzz, prompt_toolkit, fzf.

from __future__ import annotations
import argparse, json, os, re, sys, hashlib, shutil, subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# ---------- Optional niceties ----------
try:
    from rich.console import Console
    from rich.table import Table
    _HAS_RICH = True
    console = Console()
except Exception:
    _HAS_RICH = False

try:
    from rapidfuzz import fuzz as rf_fuzz
    _HAS_RAPID = True
except Exception:
    _HAS_RAPID = False

# ---------- Config ----------
CONFIG_PATH = Path.home() / ".config" / "drop" / "config.txt"

def _load_config() -> dict:
    cfg: dict = {}
    try:
        txt = CONFIG_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return cfg
    for raw in txt.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip().lower()] = v.strip()
    return cfg

_CFG = _load_config()
DATA_FILE = Path(_CFG.get("data_file_path", str(Path.home() / "Drop" / "drops.jsonl"))).expanduser()
EXPORT_DIR_DEFAULT = Path(_CFG.get("export_file_path", str(Path.home() / "Drop" / "pages"))).expanduser()

# ---------- Model ----------
HASHTAG_RE = re.compile(r"#([\w\-]+)")

@dataclass
class Scrap:
    id: int
    ts: str          # ISO 8601 (UTC) string
    text: str
    tags: List[str]

    @staticmethod
    def from_obj(obj: dict) -> "Scrap":
        return Scrap(
            id=int(obj["id"]),
            ts=str(obj["ts"]),
            text=str(obj["text"]),
            tags=list(obj.get("tags", [])),
        )

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False) + "\n"

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _to_local(ts_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return ts_iso

def _local_date_str(ts_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d")
    except Exception:
        return ts_iso[:10]

def _local_time_str(ts_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%H:%M")
    except Exception:
        return "??:??"

# ---------- Store ----------
class ScrapStore:
    def __init__(self, path: Path):
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

        self._deleted = set()
        self._next_id = 1
        self._bootstrap_from_disk()
        self._fh = open(self.path, "a", encoding="utf-8")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def _bootstrap_from_disk(self) -> None:
        max_id = 0
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                    if isinstance(obj, dict) and obj.get("op") == "del":
                        try:
                            self._deleted.add(int(obj["id"]))
                        except Exception:
                            pass
                        continue
                    try:
                        s = Scrap.from_obj(obj)
                        if s.id > max_id:
                            max_id = s.id
                    except Exception:
                        continue
        except FileNotFoundError:
            pass
        self._next_id = max_id + 1

    def _write_line(self, line: str) -> None:
        self._fh.write(line)
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def add(self, text: str) -> Scrap:
        tags = sorted({m.group(1).lower() for m in HASHTAG_RE.finditer(text)})
        scrap = Scrap(id=self._next_id, ts=_utc_now_iso(), text=text, tags=tags)
        self._next_id += 1
        self._write_line(scrap.to_jsonl())
        return scrap

    def delete(self, scrap_id: int) -> bool:
        if scrap_id in self._deleted:
            return False
        exists_live = any(s.id == scrap_id for s in self.iter_live())
        if not exists_live:
            return False
        tomb = {"op": "del", "id": scrap_id, "ts": _utc_now_iso()}
        self._write_line(json.dumps(tomb, ensure_ascii=False) + "\n")
        self._deleted.add(scrap_id)
        return True

    def iter_all(self) -> Iterable[Scrap]:
        """All non-tombstone scraps ever written (includes ones later deleted if you don't check _deleted)."""
        with self.path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("op") == "del":
                    continue
                try:
                    yield Scrap.from_obj(obj)
                except Exception:
                    continue

    def iter_live(self) -> Iterable[Scrap]:
        deleted = self._deleted
        for s in self.iter_all():
            if s.id not in deleted:
                yield s

    # Queries
    def last_n(self, n: int = 10) -> List[Scrap]:
        items = list(self.iter_live())
        return items[-n:]

    def search_substring(self, needle: str) -> List[Scrap]:
        n = needle.lower()
        hits = [s for s in self.iter_live() if n in s.text.lower()]
        # Optional: rank by token_set_ratio if available
        if _HAS_RAPID:
            hits.sort(key=lambda s: rf_fuzz.token_set_ratio(needle, s.text), reverse=True)
        else:
            # keep chronological-ish by timestamp naturally
            pass
        return hits

    def search_tag(self, tag: str) -> List[Scrap]:
        t = tag.lower().lstrip("#")
        return [s for s in self.iter_live() if t in s.tags]

# ---------- Pretty printing ----------
def _print_scrap(s: Scrap) -> None:
    ts_local = _to_local(s.ts)
    tag_str = (" #" + " #".join(s.tags)) if s.tags else ""
    print(f"[{s.id}] {ts_local}{tag_str}\n    {s.text}")

def _print_rows(headers: Tuple[str, ...], rows: List[Tuple[str, ...]]) -> None:
    if _HAS_RICH:
        table = Table(show_header=True, header_style="bold", expand=True, pad_edge=False)
        for h in headers:
            table.add_column(h, overflow="fold", no_wrap=False, justify="left")
        for r in rows:
            table.add_row(*[str(c) for c in r])
        console.print(table)
    else:
        print("\t".join(headers))
        for r in rows:
            print("\t".join(str(c) for c in r))

# ---------- Export ----------
def export_daily_markdown(store: ScrapStore, out_dir: Path, days_ago: int = 0) -> Path:
    """
    Export all live scraps from a given local calendar day to Markdown.
    days_ago=0 => today, 1 => yesterday, etc.
    """
    target_day = (date.today() - timedelta(days=days_ago)).isoformat()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = [s for s in store.iter_live() if _local_date_str(s.ts) == target_day]
    daily.sort(key=lambda s: s.ts)

    lines = [f"- Drops from [[{target_day}]]"]
    for s in daily:
        t = _local_time_str(s.ts)
        lines.append(f"\t- ({t}) {s.text}")
    lines.append("")
    out_path = out_dir / f"drop-{target_day}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _prompt_multiline(label: str = "add> ") -> Optional[str]:
    """Multiline input via prompt_toolkit if available; else single-line input."""
    # prompt_toolkit path
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.document import Document
        from prompt_toolkit.validation import Validator
        from prompt_toolkit.shortcuts import PromptSession
        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()
        @kb.add("c-enter")
        @kb.add("s-enter")  # shift-enter
        def _(event):
            event.current_buffer.insert_text("\n")

        session = PromptSession()
        text = session.prompt(label, multiline=True, key_bindings=kb)
        return text.strip() if text and text.strip() else None
    except Exception:
        pass
    # fallback: single line
    try:
        line = input(label).strip()
        return line if line else None
    except (EOFError, KeyboardInterrupt):
        return None
    

# sub-REPL

def _apply_find_filters(store: "ScrapStore", tags: List[str], substrs: List[str]) -> List["Scrap"]:
    hits: List[Scrap] = []
    tset = set(t.lower().lstrip("#") for t in tags)
    ss = [s.lower() for s in substrs]
    for s in store.iter_live():
        if tset and not tset.issubset(set(s.tags)):
            continue
        ok = True
        for sub in ss:
            if sub not in s.text.lower():
                ok = False; break
        if ok:
            hits.append(s)
    # Optional: stable ranking
    return hits

def find_subrepl(store: "ScrapStore", *, seed_all: bool, seed_text: Optional[str], confirm_deletes: bool) -> None:
    tags: List[str] = []
    substrs: List[str] = []
    if seed_all and not seed_text:
        current = list(store.iter_live())
    else:
        current = store.search_substring(seed_text) if seed_text else list(store.iter_live())

    def refresh():
        nonlocal current
        current = _apply_find_filters(store, tags, substrs)

    def show(rows_max: int = 50):
        if not current:
            print("(no matches)"); return
        rows = [(s.id, _to_local(s.ts), s.text[:140].replace("\n"," ")) for s in current[-rows_max:]]
        _print_rows(("id","time","text"), [(str(i), t, txt) for i,t,txt in rows])

    picked_id: Optional[int] = None  # `$` variable

    print("find-mode: t <tag> | untag <tag> | s <text> | uns <text> | tags | subs | ls | pick [q] | open <id|$> | del <id|$> | clear | json | back")
    while True:
        try:
            line = input("find> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in {"back", "q"}:
            break
        elif cmd == "t":
            if not arg: print("usage: t <tag>"); continue
            tags.append(arg.lstrip("#").lower()); refresh(); print(f"(tags={tags}, subs={substrs}, hits={len(current)})")
        elif cmd == "untag":
            if not arg: print("usage: untag <tag>"); continue
            t = arg.lstrip("#").lower()
            tags[:] = [x for x in tags if x != t]; refresh(); print(f"(tags={tags}, subs={substrs}, hits={len(current)})")
        elif cmd == "s":
            if not arg: print("usage: s <text>"); continue
            substrs.append(arg.lower()); refresh(); print(f"(tags={tags}, subs={substrs}, hits={len(current)})")
        elif cmd == "uns":
            if not arg: print("usage: uns <text>"); continue
            d = arg.lower()
            substrs[:] = [x for x in substrs if x != d]; refresh(); print(f"(tags={tags}, subs={substrs}, hits={len(current)})")
        elif cmd == "tags":
            print("tags:", tags if tags else "(none)")
        elif cmd == "subs":
            print("subs:", substrs if substrs else "(none)")
        elif cmd == "ls":
            show()
        elif cmd == "pick":
            seed = arg or ""
            cands = current
            if seed:
                cands = [s for s in current if seed.lower() in s.text.lower()]
            if not cands:
                print("(no candidates)"); continue
            items = [(s.id, s.text[:140].replace("\n"," ")) for s in cands]
            sid = _picker(items, prompt_text="pick> ")
            if sid is not None:
                picked_id = sid
                print(f"picked ${picked_id}")
        elif cmd == "open":
            token = arg.strip()
            if token == "$":
                if picked_id is None: print("(no $ set)"); continue
                sid = picked_id
            else:
                if not token: print("usage: open <id|$>"); continue
                try:
                    sid = int(token)
                except ValueError:
                    print("id must be an integer"); continue
            one = next((s for s in current if s.id == sid), None)
            if not one: print("not found in current set"); continue
            _print_scrap(one)
        elif cmd == "del":
            token = arg.strip()
            if token == "$":
                if picked_id is None: print("(no $ set)"); continue
                sid = picked_id
            else:
                if not token: print("usage: del <id|$>"); continue
                try:
                    sid = int(token)
                except ValueError:
                    print("id must be an integer"); continue
            if confirm_deletes:
                ans = input(f"delete {sid}? [y/N] ").strip().lower()
                if ans not in {"y","yes"}:
                    print("(canceled)"); continue
            ok = store.delete(sid)
            print("deleted" if ok else "no such live id")
            refresh()
        elif cmd == "clear":
            tags.clear(); substrs.clear(); refresh(); print("(filters cleared)")
        elif cmd == "json":
            print(json.dumps([asdict(s) for s in current], ensure_ascii=False))
        else:
            print("unknown command")


# main REPL

def repl_main(store: "ScrapStore", *, confirm_deletes: bool) -> None:
    print("Drop REPL — type 'help' for commands. Ctrl-D to quit.")
    # Dashboard
    items = store.last_n(10)
    if items:
        rows = [(s.id, _to_local(s.ts), s.text[:120].replace("\n"," ")) for s in items]
        _print_rows(("id","time","text"), [(str(i), t, txt) for i,t,txt in rows])
    else:
        print("(no drops yet)")

    while True:
        try:
            line = input("hdp> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in {"q","quit","exit"}:
            break
        elif cmd in {"?","help"}:
            print("commands: add|a [text] • ls [N] • search <text> • tag <tag> • open <id> • del <id> • export [--days-ago N] [--path DIR] • find [--all] [seed] • help • quit")
        elif cmd in {"a","add"}:
            text = arg if arg else _prompt_multiline("add> ")
            if not text:
                print("(nothing to add)"); continue
            s = store.add(text)
            sys.stderr.write(f"[saved] [{s.id}] {_to_local(s.ts)} — {s.text[:80].replace(os.linesep, ' ')}\n")
            print(s.id)
        elif cmd == "ls":
            n = 10
            if arg:
                try: n = max(1, int(arg))
                except ValueError: print("usage: ls [N]"); continue
            items = store.last_n(n)
            if not items: print("(no drops)"); continue
            rows = [(s.id, _to_local(s.ts), s.text[:120].replace("\n"," ")) for s in items]
            _print_rows(("id","time","text"), [(str(i), t, txt) for i,t,txt in rows])
        elif cmd == "search":
            if not arg: print("usage: search <text>"); continue
            hits = store.search_substring(arg)
            if not hits: print("(no matches)"); continue
            for s in hits: _print_scrap(s)
        elif cmd == "tag":
            if not arg: print("usage: tag <tag>"); continue
            hits = store.search_tag(arg)
            if not hits: print("(no matches)"); continue
            for s in hits: _print_scrap(s)
        elif cmd == "open":
            if not arg: print("usage: open <id>"); continue
            try: sid = int(arg)
            except ValueError: print("id must be an integer"); continue
            one = next((s for s in store.iter_live() if s.id == sid), None)
            if not one: print("not found"); continue
            _print_scrap(one)
        elif cmd == "del":
            if not arg: print("usage: del <id>"); continue
            try: sid = int(arg)
            except ValueError: print("id must be an integer"); continue
            if confirm_deletes:
                ans = input(f"delete {sid}? [y/N] ").strip().lower()
                if ans not in {"y","yes"}:
                    print("(canceled)"); continue
            ok = store.delete(sid)
            print("deleted" if ok else "no such live id")
        elif cmd == "export":
            # parse tiny inline flags: --days-ago N, --path DIR
            days_ago, out_dir = 0, None
            if arg:
                toks = arg.split()
                i = 0
                while i < len(toks):
                    t = toks[i]
                    if t == "--days-ago" and i + 1 < len(toks):
                        try: days_ago = int(toks[i+1]); i += 2; continue
                        except ValueError: print("export: --days-ago N"); break
                    elif t == "--path" and i + 1 < len(toks):
                        out_dir = Path(toks[i+1]); i += 2; continue
                    else:
                        print("export: usage: export [--days-ago N] [--path DIR]"); break
            out_dir = out_dir if out_dir else EXPORT_DIR_DEFAULT
            pth = export_daily_markdown(store, out_dir, days_ago=max(0, days_ago))
            target_day = (date.today() - timedelta(days=max(0, days_ago))).isoformat()
            count = sum(1 for s in store.iter_live() if _local_date_str(s.ts) == target_day)
            print(f"Exported {count} drops to {pth}")
        elif cmd == "find":
            # parse: find [--all] [seed text...]
            seed_all = False
            seed = ""
            if arg:
                if arg.startswith("--all"):
                    seed_all = True
                    seed = arg[len("--all"):].strip()
                else:
                    seed = arg
            find_subrepl(store, seed_all=seed_all, seed_text=(seed or None), confirm_deletes=confirm_deletes)
        else:
            print("unknown command; type 'help'")

# ---------- Interactive find ----------
def _picker(items: List[Tuple[int, str]], prompt_text: str = "Select> ") -> Optional[int]:
    # 1) prompt_toolkit
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
        display = [f"{title}  [{sid}]" for sid, title in items]
        comp = FuzzyCompleter(WordCompleter(display, sentence=True))
        choice = (prompt(prompt_text, completer=comp) or "").strip()
        if not choice:
            return None
        if choice in display:
            return items[display.index(choice)][0]
        # fuzzy fallback: pick first contains match
        for sid, title in items:
            if choice.lower() in title.lower():
                return sid
    except Exception:
        pass
    # 2) fzf
    if shutil.which("fzf"):
        lines = [f"{sid}\t{title}" for sid, title in items]
        try:
            proc = subprocess.Popen(
                ["fzf", "--with-nth=2..", "--prompt", prompt_text],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            out, _ = proc.communicate("\n".join(lines))
            out = (out or "").strip()
            if not out:
                return None
            return int(out.split("\t", 1)[0])
        except Exception:
            pass
    # 3) numbered fallback
    for i, (sid, title) in enumerate(items[:50], 1):
        print(f"{i:2d}. {title}  [{sid}]")
    try:
        sel = input("Number (blank to cancel): ").strip()
        if not sel:
            return None
        idx = int(sel) - 1
        if 0 <= idx < len(items):
            return items[idx][0]
    except Exception:
        return None
    return None

def interactive_narrow(store: ScrapStore) -> None:
    """
    Minimal interactive narrower: supports refining and light actions.
    commands:
      t <tag>        add tag filter (repeatable)
      s <text>       add substring filter (case-insensitive)
      ls             list current hits
      open <id>      show one item
      del <id>       delete (tombstone) one item
      clear          reset filters to all live
      q              quit
    """
    tags: List[str] = []
    substr: Optional[str] = None
    current: List[Scrap] = list(store.iter_live())

    def apply() -> List[Scrap]:
        out = []
        for s in store.iter_live():
            if tags and not set(tags).issubset(set(s.tags)):
                continue
            if substr and substr.lower() not in s.text.lower():
                continue
            out.append(s)
        return out

    print("Interactive find. Commands: t <tag> | s <text> | ls | open <id> | del <id> | clear | q")
    while True:
        try:
            line = input("find> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        if cmd == "q":
            break
        elif cmd == "clear":
            tags, substr = [], None
            current = list(store.iter_live())
            print("(filters cleared)")
        elif cmd == "t":
            if not arg:
                print("usage: t <tag>")
                continue
            t = arg.lstrip("#").lower()
            tags.append(t)
            current = apply()
            print(f"(tags={tags}, substr={substr!r}, hits={len(current)})")
        elif cmd == "s":
            if not arg:
                print("usage: s <text>")
                continue
            substr = arg
            current = apply()
            print(f"(tags={tags}, substr={substr!r}, hits={len(current)})")
        elif cmd == "ls":
            if not current:
                print("(no matches)")
            else:
                rows = [(s.id, _to_local(s.ts), s.text[:120].replace("\n", " ")) for s in current[-50:]]
                _print_rows(("id","time","text"), [(str(i), t, txt) for i,t,txt in rows])
        elif cmd == "open":
            if not arg:
                print("usage: open <id>"); continue
            try:
                sid = int(arg)
            except ValueError:
                print("id must be an integer"); continue
            one = next((s for s in current if s.id == sid), None)
            if not one:
                print("not found in current set")
            else:
                _print_scrap(one)
        elif cmd == "del":
            if not arg:
                print("usage: del <id>"); continue
            try:
                sid = int(arg)
            except ValueError:
                print("id must be an integer"); continue
            ok = store.delete(sid)
            print("deleted" if ok else "no such live id")
            current = apply()
        else:
            print("unknown command")

# ---------- CLI ----------
def _print_capture_result(scrap: Scrap, id_only: bool) -> None:
    # ID to stdout always (pipe-friendly)
    print(scrap.id)
    if not id_only:
        # Friendly summary to stderr
        sys.stderr.write(f"[saved] [{scrap.id}] {_to_local(scrap.ts)} — {scrap.text[:80].replace(os.linesep, ' ')}\n")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hdp", description="Drop — tiny personal drop-capture & search")
    sub = p.add_subparsers(dest="cmd")

    # Default capture via positional at top-level:
    p.add_argument("text", nargs="?", help="Capture a drop (default action). Same as: hdp add \"...\"")
    p.add_argument("--id-only", action="store_true", help="On capture, print only the ID (suppress summary)")
    # Subcommands
    sp = sub.add_parser("add", help="Add a new drop (alias of default capture)")
    sp.add_argument("text")
    sp.add_argument("--id-only", action="store_true")
    sp.set_defaults(func="add")

    sp = sub.add_parser("list", help="List recent drops")
    sp.add_argument("--n", type=int, default=10)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func="list")

    sp = sub.add_parser("search", help="Search by substring (case-insensitive)")
    sp.add_argument("query")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func="search")

    sp = sub.add_parser("tag", help="Search by tag (with or without leading #)")
    sp.add_argument("tag")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func="tag")

    sp = sub.add_parser("find", help="Find interactively or pick a single ID")
    sp.add_argument("query", nargs="?", help="Seed text query (optional)")
    sp.add_argument("-i", "--interactive", action="store_true", help="Interactive narrower")
    sp.add_argument("-p", "--pick", action="store_true", help="Picker: return a single ID to stdout")
    sp.add_argument("--all", action="store_true", help="Ignore query and start from all live drops")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func="find")

    sp = sub.add_parser("delete", help="Delete (tombstone) a drop by id")
    sp.add_argument("id", type=int)
    sp.set_defaults(func="delete")

    sp = sub.add_parser("export", help="Export a day’s drops to Markdown (Logseq-friendly)")
    sp.add_argument("--days-ago", type=int, default=0, help="0=today, 1=yesterday, ...")
    sp.add_argument("--path", type=Path, help="Export directory (overrides config export_file_path)")
    sp.set_defaults(func="export")

    sp = sub.add_parser("repl", help="Interactive REPL (stateless main; stateful find-mode)")
    sp.add_argument("--no-confirm", action="store_true", help="Don’t ask for confirmation on deletes")
    sp.set_defaults(func="repl")

    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # If a subcommand is specified, run that; else, if positional 'text' is provided, capture; else help.
    if not getattr(args, "cmd", None) and not getattr(args, "text", None):
        parser.print_help()
        return

    store = ScrapStore(DATA_FILE)
    try:
        cmd = getattr(args, "cmd", None)

        # Default capture (no subcommand)
        if cmd is None and args.text:
            scrap = store.add(args.text)
            _print_capture_result(scrap, id_only=args.id_only)
            return

        if cmd == "add":
            scrap = store.add(args.text)
            _print_capture_result(scrap, id_only=args.id_only)
            return

        if cmd == "list":
            items = store.last_n(max(1, args.n))
            if args.json:
                print(json.dumps([asdict(s) for s in items], ensure_ascii=False))
            else:
                if not items:
                    print("No drops yet.")
                else:
                    rows = [(s.id, _to_local(s.ts), (s.text[:120].replace("\n"," "))) for s in items]
                    _print_rows(("id","time","text"), [(str(i), t, txt) for i,t,txt in rows])
            return

        if cmd == "search":
            hits = store.search_substring(args.query)
            if args.json:
                print(json.dumps([asdict(s) for s in hits], ensure_ascii=False))
            else:
                if not hits:
                    print("No matches.")
                else:
                    for s in hits:
                        _print_scrap(s)
            return
    
        if cmd == "repl":
            repl_main(store, confirm_deletes=not args.no_confirm)
            return

        if cmd == "tag":
            hits = store.search_tag(args.tag)
            if args.json:
                print(json.dumps([asdict(s) for s in hits], ensure_ascii=False))
            else:
                if not hits:
                    print("No matches.")
                else:
                    for s in hits:
                        _print_scrap(s)
            return

        if cmd == "find":
            seed_all = args.all or not args.query
            if args.pick:
                # Build candidate list
                cands = list(store.iter_live()) if seed_all else store.search_substring(args.query)
                if not cands:
                    sys.exit(1)
                items = [(s.id, s.text[:140].replace("\n"," ")) for s in cands]
                sid = _picker(items)
                if sid is not None:
                    print(sid)
                    sys.exit(0)
                else:
                    sys.exit(2)
            if args.interactive:
                interactive_narrow(store)
                return
            # non-interactive, one-shot
            hits = list(store.iter_live()) if seed_all else store.search_substring(args.query)
            if args.json:
                print(json.dumps([asdict(s) for s in hits], ensure_ascii=False))
            else:
                if not hits:
                    print("No matches.")
                else:
                    for s in hits:
                        _print_scrap(s)
            return

        if cmd == "delete":
            ok = store.delete(args.id)
            if ok:
                print(f"Deleted {args.id}.")
            else:
                print(f"No live drop with id {args.id}.")
            return

        if cmd == "export":
            out_dir = args.path if args.path else EXPORT_DIR_DEFAULT
            out_path = export_daily_markdown(store, out_dir, days_ago=max(0, args.days_ago))
            # Summary
            target_day = (date.today() - timedelta(days=max(0, args.days_ago))).isoformat()
            count = sum(1 for s in store.iter_live() if _local_date_str(s.ts) == target_day)
            print(f"Exported {count} drops to {out_path}")
            return

        # Fallback: shouldn’t reach here
        parser.print_help()

    finally:
        store.close()

if __name__ == "__main__":
    main()
