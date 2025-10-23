#!/usr/bin/env python3
# drop-repl.py — “Drop” (drp/drop)
# Default: one-shot capture; subcommands for everything else.
# Python 3.9+. No hard deps; optional: rich, rapidfuzz, prompt_toolkit, fzf, pyyaml (for outflows).

from __future__ import annotations
import argparse, json, os, re, sys, hashlib, shutil, subprocess, tempfile, random, time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

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

try:
    import yaml  # PyYAML (optional)
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# ---------- Config ----------
CONFIG_DIR = Path.home() / ".config" / "drop"
CONFIG_PATH = CONFIG_DIR / "config.txt"
OUTFLOWS_PATH = CONFIG_DIR / "outflows.yaml"

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
_EDITOR_CMD = _CFG.get("editor_cmd", "").strip() or os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"

# ---------- ULID (no external dep) ----------
# ULID: 48-bit ms timestamp + 80-bit randomness, Crockford base32 (26 chars), lexicographically sortable by time.
_CROCK = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

def _to_crockford(num: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_CROCK[num & 31])
        num >>= 5
    return "".join(reversed(out))

def new_ulid() -> str:
    t_ms = int(time.time() * 1000)
    # 48-bit time
    time_part = _to_crockford(t_ms, 10)  # 48 bits fit into 10 base32 chars
    # 80-bit randomness => 16 base32 chars
    rand_hi = random.getrandbits(80)
    rand_part = _to_crockford(rand_hi, 16)
    return time_part + rand_part  # 26 chars

def _resolve_unique_id(store: "ScrapStore", token: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve a user-supplied ID or prefix to a unique full id among *live* notes.
    Returns (full_id, error_message). If error_message is not None, resolution failed.
    """
    token = (token or "").strip()
    if not token:
        return None, "no id provided"
    # exact match wins
    for s in store.iter_live():
        if s.id == token:
            return s.id, None
    # prefix match
    cands = [s.id for s in store.iter_live() if s.id.startswith(token)]
    if not cands:
        return None, f"no live note matches id/prefix '{token}'"
    if len(cands) > 1:
        # show 2 suggestions to help the user disambiguate
        hints = ", ".join(c[:12] for c in cands[:2])
        return None, f"ambiguous id/prefix '{token}' ({len(cands)} matches; try a longer prefix like: {hints})"
    return cands[0], None


# ---------- Model ----------
HASHTAG_RE = re.compile(r"#([\w\-]+)")

@dataclass
class Scrap:
    id: str          # ULID/string; legacy ints are normalized to str
    ts: str          # ISO 8601 (UTC) string
    text: str
    tags: List[str]

    @staticmethod
    def from_obj(obj: dict) -> "Scrap":
        # normalize id to string
        rid = obj.get("id")
        if isinstance(rid, int):
            rid = str(rid)
        elif not isinstance(rid, str):
            rid = str(rid)
        return Scrap(
            id=rid,
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
        self._bootstrap_from_disk()
        self._fh = open(self.path, "a", encoding="utf-8")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def _bootstrap_from_disk(self) -> None:
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
                        rid = obj.get("id")
                        self._deleted.add(str(rid))
                        continue
        except FileNotFoundError:
            pass

    def _write_line(self, line: str) -> None:
        self._fh.write(line)
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def add(self, text: str) -> Scrap:
        tags = sorted({m.group(1).lower() for m in HASHTAG_RE.finditer(text)})
        scrap = Scrap(id=new_ulid(), ts=_utc_now_iso(), text=text, tags=tags)
        self._write_line(scrap.to_jsonl())
        return scrap

    def delete(self, scrap_id: str) -> bool:
        sid = str(scrap_id)
        exists_live = any(s.id == sid for s in self.iter_live())
        if not exists_live:
            return False
        tomb = {"op": "del", "id": sid, "ts": _utc_now_iso()}
        self._write_line(json.dumps(tomb, ensure_ascii=False) + "\n")
        self._deleted.add(sid)
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
                    s = Scrap.from_obj(obj)
                    if s.id not in self._deleted:
                        yield s
                except Exception:
                    continue

    def iter_live(self) -> Iterable[Scrap]:
        yield from self.iter_all()

    # Queries
    def last_n(self, n: int = 10) -> List[Scrap]:
        items = list(self.iter_live())
        return items[-n:]

    def search_substring(self, needle: str) -> List[Scrap]:
        n = needle.lower()
        hits = [s for s in self.iter_live() if n in s.text.lower()]
        if _HAS_RAPID:
            hits.sort(key=lambda s: rf_fuzz.token_set_ratio(needle, s.text), reverse=True)
        return hits

    def search_tag(self, tag: str) -> List[Scrap]:
        t = tag.lower().lstrip("#")
        return [s for s in self.iter_live() if t in s.tags]

# ---------- Pretty printing ----------
def _short_id(sid: str, n: int = 10) -> str:
    return sid[:n]

def _print_scrap(s: Scrap) -> None:
    ts_local = _to_local(s.ts)
    tag_str = (" #" + " #".join(s.tags)) if s.tags else ""
    print(f"[{_short_id(s.id)}] {ts_local}{tag_str}\n    {s.text}")

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

# ---------- Outflows (export-only) ----------
@dataclass
class Outflow:
    name: str
    rule: str           # "*", "#tag", or "@route"
    export_path: Path
    elide_rule_token: bool = False  # renamed from elide_rule_tag; we still read the old key

def _load_outflows() -> List[Outflow]:
    if not OUTFLOWS_PATH.exists():
        return []
    if not _HAS_YAML:
        sys.stderr.write("[warn] outflows.yaml present but PyYAML not installed; ignoring outflows.\n")
        return []
    try:
        data = yaml.safe_load(OUTFLOWS_PATH.read_text(encoding="utf-8")) or []
        out: List[Outflow] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", f"outflow_{i}"))
            rule = str(item.get("rule", "*")).strip()
            export_path = Path(str(item.get("export_path", ""))).expanduser()
            if not str(export_path):
                sys.stderr.write(f"[warn] outflow '{name}' missing export_path; skipped.\n")
                continue
            # accept either elide_rule_token (new) or elide_rule_tag (legacy)
            elide = bool(item.get("elide_rule_token", item.get("elide_rule_tag", False)))
            out.append(Outflow(name=name, rule=rule, export_path=export_path, elide_rule_token=elide))
        return out
    except Exception as e:
        sys.stderr.write(f"[warn] Failed to parse outflows.yaml: {e}\n")
        return []

def _matches_outflow(s: Scrap, of: Outflow) -> bool:
    r = of.rule.strip()
    if r == "*":
        return True
    if r.startswith("#"):
        wanted = r[1:].lower()
        return wanted in s.tags
    if r.startswith("@"):
        route = r[1:].lower()
        return _contains_route_token(s.text, route)
    # bare tag fallback
    return r.lower().lstrip("#") in s.tags


# ---- Routing (@route) + leading-only elision helpers ----
def _contains_route_token(text: str, route: str) -> bool:
    """
    Does text contain '@route' as a standalone token (not inside a word)?
    """
    pattern = re.compile(rf"(?<![\w\-])@{re.escape(route)}(?![\w\-])")
    return bool(pattern.search(text))

def _elide_leading_tag(text: str, tag: str) -> str:
    """
    Remove a single leading '#tag' token (and following spaces) if present.
    """
    pattern = re.compile(rf"^\s*#(?i:{re.escape(tag)})(?![\w\-])[ \t]*")
    return pattern.sub("", text, count=1)

def _elide_leading_route(text: str, route: str) -> str:
    """
    Remove a single leading '@route' token (and following spaces) if present.
    """
    pattern = re.compile(rf"^\s*@(?i:{re.escape(route)})(?![\w\-])[ \t]*")
    return pattern.sub("", text, count=1)

def _render_text_for_outflow(s: Scrap, of: Outflow) -> str:
    if not of.elide_rule_token:
        return s.text
    r = of.rule.strip()
    # Only elide if rule is a token rule and matches
    if r.startswith("#"):
        tag = r[1:].lower()
        if tag in s.tags:
            return _elide_leading_tag(s.text, tag)
        return s.text
    if r.startswith("@"):
        route = r[1:].lower()
        if _contains_route_token(s.text, route):
            return _elide_leading_route(s.text, route)
        return s.text
    return s.text

def export_daily_markdown(store: ScrapStore, out_dir: Path, days_ago: int = 0,
                          filter_fn=None, transform_fn=None) -> Path:
    """
    Export all live scraps from a given local calendar day to Markdown.
    Optionally filter_fn(Scrap)->bool and transform_fn(Scrap)->str (text override).
    """
    target_day = (date.today() - timedelta(days=days_ago)).isoformat()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = [s for s in store.iter_live() if _local_date_str(s.ts) == target_day]
    if filter_fn:
        daily = [s for s in daily if filter_fn(s)]
    daily.sort(key=lambda s: s.ts)

    lines = [f"- Drops from [[{target_day}]]"]
    for s in daily:
        t = _local_time_str(s.ts)
        text = transform_fn(s) if transform_fn else s.text
        lines.append(f"\t- ({t}) {text}")
    lines.append("")
    out_path = out_dir / f"drop-{target_day}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path

# ---------- Editor capture (same as before) ----------
def _resolve_editor_argv(tmp_path: Path) -> List[str]:
    cmd = _EDITOR_CMD
    if "{file}" in cmd:
        replaced = cmd.replace("{file}", str(tmp_path))
        return ["/bin/sh", "-c", replaced]
    else:
        parts = cmd.split()
        if not parts:
            parts = ["vi"]
        return parts + [str(tmp_path)]

def _ensure_tempdir() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    d = cache_root / "drop" / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d

def capture_via_editor(seed: Optional[str] = None, *, keep_temp: bool = False) -> Tuple[Optional[str], bool]:
    tmpdir = _ensure_tempdir()
    fd, tmppath = tempfile.mkstemp(prefix="drop-", suffix=".md", dir=str(tmpdir), text=True)
    os.close(fd)
    p = Path(tmppath)
    try:
        if seed:
            p.write_text(seed, encoding="utf-8")
        argv = _resolve_editor_argv(p)
        try:
            proc = subprocess.run(argv)
        except FileNotFoundError:
            sys.stderr.write(f"[error] Editor not found: {_EDITOR_CMD}\n")
            return None, True
        except Exception as e:
            sys.stderr.write(f"[error] Failed to launch editor: {e}\n")
            return None, True
        if proc.returncode != 0:
            return None, True
        try:
            content = p.read_text(encoding="utf-8")
        except Exception as e:
            sys.stderr.write(f"[error] Could not read editor temp file: {e}\n")
            return None, True
        content = content.strip("\n\r\t ")
        if content == "":
            return None, False
        return content, False
    finally:
        if not keep_temp:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

def _prompt_multiline(label: str = "add> ") -> Optional[str]:
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.shortcuts import PromptSession
        from prompt_toolkit.key_binding import KeyBindings
        kb = KeyBindings()
        @kb.add("c-enter")
        @kb.add("s-enter")
        def _(event):
            event.current_buffer.insert_text("\n")
        session = PromptSession()
        text = session.prompt(label, multiline=True, key_bindings=kb)
        return text.strip() if text and text.strip() else None
    except Exception:
        pass
    try:
        line = input(label).strip()
        return line if line else None
    except (EOFError, KeyboardInterrupt):
        return None

# ---------- sub-REPL (find) ----------
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
        rows = [( _short_id(s.id), _to_local(s.ts), s.text[:140].replace("\n"," ")) for s in current[-rows_max:]]
        _print_rows(("id","time","text"), [(str(i), t, txt) for i,t,txt in rows])

    picked_id: Optional[str] = None  # `$` variable

    help_string = "find-mode: h | t <tag> | untag <tag> | s <text> | uns <text> | tags | subs | ls | pick [q] | open <id|$> | del <id|$> | clear | json | back"
    print(help_string)
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
        elif cmd == "h":
            print(help_string)
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
            cands = current if not seed else [s for s in current if seed.lower() in s.text.lower()]
            if not cands:
                print("(no candidates)"); continue
            items = [(s.id, s.text[:140].replace("\n"," ")) for s in cands]
            sid = _picker(items, prompt_text="pick> ")
            if sid is not None:
                picked_id = sid
                print(f"picked ${picked_id}")
        # Interactive find subrepl
        elif cmd == "open":
            token = arg.strip()
            if token == "$":
                if picked_id is None: print("(no $ set)"); continue
                sid = picked_id
                one = next((s for s in current if s.id == sid), None)
            else:
                full_id, err = _resolve_unique_id(store, token)
                if err: print(err); continue
                one = next((s for s in store.iter_live() if s.id == full_id), None)
            if not one: print("not found in current set"); continue
            _print_scrap(one)
        # Interactive find subrepl
        elif cmd == "del":
            token = arg.strip()
            if token == "$":
                if picked_id is None: print("(no $ set)"); continue
                full_id = picked_id
            else:
                full_id, err = _resolve_unique_id(store, token)
                if err: print(err); continue
            if confirm_deletes:
                ans = input(f"delete {full_id}? [y/N] ").strip().lower()
                if ans not in {"y","yes"}:
                    print("(canceled)"); continue
            ok = store.delete(full_id)
            print("deleted" if ok else "no such live id")
            refresh()
        elif cmd == "clear":
            tags.clear(); substrs.clear(); refresh(); print("(filters cleared)")
        elif cmd == "json":
            print(json.dumps([asdict(s) for s in current], ensure_ascii=False))
        else:
            print("unknown command")

# ---------- Interactive find ----------
def _picker(items: List[Tuple[str, str]], prompt_text: str = "Select> ") -> Optional[str]:
    # 1) prompt_toolkit
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
        display = [f"{title}  [{_short_id(sid)}]" for sid, title in items]
        comp = FuzzyCompleter(WordCompleter(display, sentence=True))
        choice = (prompt(prompt_text, completer=comp) or "").strip()
        if not choice:
            return None
        if choice in display:
            idx = display.index(choice)
            return items[idx][0]
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
            return out.split("\t", 1)[0]
        except Exception:
            pass
    # 3) numbered fallback
    for i, (sid, title) in enumerate(items[:50], 1):
        print(f"{i:2d}. {title}  [{_short_id(sid)}]")
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

# ---------- CLI ----------
def _print_capture_result(scrap: Scrap, id_only: bool) -> None:
    print(scrap.id)
    if not id_only:
        sys.stderr.write(f"[saved] [{_short_id(scrap.id)}] {_to_local(scrap.ts)} — {scrap.text[:80].replace(os.linesep, ' ')}\n")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hdp", description="Drop — tiny personal drop-capture & search")
    sub = p.add_subparsers(dest="cmd")

    # Default capture via positional at top-level:
    p.add_argument("text", nargs="?", help="Capture a drop (default action). Same as: hdp add \"...\"")
    p.add_argument("--id-only", action="store_true", help="On capture, print only the ID (suppress summary)")

    # Subcommands
    sp = sub.add_parser("add", help="Add a new drop (alias of default capture)")
    sp.add_argument("text", nargs="?", help="Optional prefill when using --editor")
    sp.add_argument("-e", "--editor", action="store_true", help="Open $VISUAL/$EDITOR (or config editor_cmd) to compose the note")
    sp.add_argument("--keep-temp", action="store_true", help="Do not delete the temporary editor file (debugging)")
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
    sp.add_argument("-i", "--interactive", action="store_true", help="Open interactive find repl")
    sp.add_argument("-p", "--pick", action="store_true", help="Picker: return a single ID to stdout")
    sp.add_argument("--all", action="store_true", help="Ignore query and start from all live drops")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func="find")

    sp = sub.add_parser("delete", help="Delete (tombstone) a drop by id or unique prefix")
    sp.add_argument("id")
    sp.set_defaults(func="delete")

    sp = sub.add_parser("export", help="Export a day’s drops to Markdown")
    sp.add_argument("--days-ago", type=int, default=0, help="0=today, 1=yesterday, ...")
    sp.add_argument("--path", type=Path, help="Export directory (overrides config export_file_path when no outflows)")
    sp.add_argument("--outflow", help="Export only the named outflow from outflows.yaml")
    sp.set_defaults(func="export")

    sp = sub.add_parser("repl", help="Interactive REPL (stateless main; stateful find-mode)")
    sp.add_argument("--no-confirm", action="store_true", help="Don’t ask for confirmation on deletes")
    sp.set_defaults(func="repl")

    return p

# ---------- REPL main ----------
def repl_main(store: "ScrapStore", *, confirm_deletes: bool) -> None:
    print("Drop REPL — type 'help' for commands. Ctrl-D to quit.")
    items = store.last_n(10)
    if items:
        rows = [(_short_id(s.id), _to_local(s.ts), s.text[:120].replace("\n"," ")) for s in items]
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
            print("commands: add|a [text] • e|edit [seed] • ls [N] • search <text> • tag <tag> • open <id> • del <id> • export [--days-ago N] [--outflow NAME] • find [--all] [seed] • help • quit")
        elif cmd in {"e","edit"}:
            seed = arg if arg else None
            text, canceled = capture_via_editor(seed, keep_temp=False)
            if canceled:
                continue
            if text is None:
                sys.stderr.write("Received no text from the editor. Did you enter anything?\n")
                continue
            s = store.add(text)
            sys.stderr.write(f"[saved] [{_short_id(s.id)}] {_to_local(s.ts)} — {s.text[:80].replace(os.linesep, ' ')}\n")
            print(s.id)
        elif cmd in {"a","add"}:
            text = arg if arg else _prompt_multiline("add> ")
            if not text:
                print("(nothing to add)"); continue
            s = store.add(text)
            sys.stderr.write(f"[saved] [{_short_id(s.id)}] {_to_local(s.ts)} — {s.text[:80].replace(os.linesep, ' ')}\n")
            print(s.id)
        elif cmd == "ls":
            n = 10
            if arg:
                try: n = max(1, int(arg))
                except ValueError: print("usage: ls [N]"); continue
            items = store.last_n(n)
            if not items: print("(no drops)"); continue
            rows = [(_short_id(s.id), _to_local(s.ts), s.text[:120].replace("\n"," ")) for s in items]
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
        # main repl
        elif cmd == "open":
            if not arg: print("usage: open <id|prefix>"); continue
            full_id, err = _resolve_unique_id(store, arg)
            if err: print(err); continue
            one = next((s for s in store.iter_live() if s.id == full_id), None)
            if not one: print("not found"); continue
            _print_scrap(one)
        #main repl
        elif cmd == "del":
            if not arg: print("usage: del <id|prefix>"); continue
            full_id, err = _resolve_unique_id(store, arg)
            if err: print(err); continue
            if confirm_deletes:
                ans = input(f"delete {full_id}? [y/N] ").strip().lower()
                if ans not in {"y","yes"}:
                    print("(canceled)"); continue
            ok = store.delete(full_id)
            print("deleted" if ok else "no such live id")
        elif cmd == "export":
            # Allow: export [--days-ago N] [--outflow NAME]
            # If outflows exist and no --path given, export all configured outflows (or the named one).
            # Else fallback to legacy single export path (or overridden --path).
            # parse tiny inline flags (legacy REPL path)
            days_ago, outflow_name, out_path_override = 0, None, None
            if arg:
                toks = arg.split()
                i = 0
                while i < len(toks):
                    t = toks[i]
                    if t == "--days-ago" and i + 1 < len(toks):
                        try: days_ago = int(toks[i+1]); i += 2; continue
                        except ValueError: print("export: --days-ago N"); break
                    elif t == "--outflow" and i + 1 < len(toks):
                        outflow_name = toks[i+1]; i += 2; continue
                    elif t == "--path" and i + 1 < len(toks):
                        out_path_override = Path(toks[i+1]); i += 2; continue
                    else:
                        print("export: usage: export [--days-ago N] [--outflow NAME] [--path DIR]"); break
            _do_export(store, days_ago=days_ago, outflow_name=outflow_name, path_override=out_path_override)
        elif cmd == "find":
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

def _do_export(store: ScrapStore, *, days_ago: int, outflow_name: Optional[str], path_override: Optional[Path]) -> None:
    outflows = _load_outflows()
    target_day = (date.today() - timedelta(days=max(0, days_ago))).isoformat()

    if outflows and not path_override:
        # Export to all or to one named outflow
        targets = outflows
        if outflow_name:
            targets = [of for of in outflows if of.name == outflow_name]
            if not targets:
                print(f"No such outflow: {outflow_name}")
                return
        total_written = 0
        for of in targets:
            of.export_path.mkdir(parents=True, exist_ok=True)
            path = export_daily_markdown(
                store,
                of.export_path,
                days_ago=max(0, days_ago),
                filter_fn=lambda s, of=of: _matches_outflow(s, of),
                transform_fn=lambda s, of=of: _render_text_for_outflow(s, of),
            )
            count = sum(1 for s in store.iter_live() if _local_date_str(s.ts) == target_day and _matches_outflow(s, of))
            total_written += count
            print(f"[{of.name}] Exported {count} drops to {path}")
        if len(targets) > 1:
            print(f"Done. {len(targets)} outflows exported.")
        return

    # Fallback: legacy single export path (or override path)
    out_dir = path_override if path_override else EXPORT_DIR_DEFAULT
    out_path = export_daily_markdown(store, out_dir, days_ago=max(0, days_ago))
    count = sum(1 for s in store.iter_live() if _local_date_str(s.ts) == target_day)
    print(f"Exported {count} drops to {out_path}")

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
            if getattr(args, "editor", False):
                text, canceled = capture_via_editor(args.text, keep_temp=getattr(args, "keep_temp", False))
                if canceled:
                    return
                if text is None:
                    sys.stderr.write("Received no text from the editor. Did you enter anything?\n")
                    sys.exit(3)
                scrap = store.add(text)
                _print_capture_result(scrap, id_only=args.id_only)
                return
            if args.text is None:
                parser.error("add requires TEXT or use --editor")
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
                    rows = [(_short_id(s.id), _to_local(s.ts), (s.text[:120].replace("\n"," "))) for s in items]
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
                find_subrepl(store, seed_all=True, seed_text=None, confirm_deletes=True)
                return
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
            full_id, err = _resolve_unique_id(store, args.id)
            if err:
                print(err); return
            ok = store.delete(full_id)
            if ok:
                print(f"Deleted {full_id}.")
            else:
                print(f"No live drop with id {full_id}.")
            return

        if cmd == "export":
            _do_export(store, days_ago=max(0, args.days_ago), outflow_name=getattr(args, "outflow", None),
                       path_override=getattr(args, "path", None))
            return

        # Fallback: shouldn’t reach here
        parser.print_help()

    finally:
        store.close()

if __name__ == "__main__":
    main()
