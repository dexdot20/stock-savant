from __future__ import annotations

import atexit
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import TextIO

from rich.console import Console

from config import get_config
from .paths import get_runtime_dir

_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_TRANSCRIPT_PATH = get_runtime_dir() / "terminal.txt"
_TRANSCRIPT_LOCK = threading.Lock()
_TRANSCRIPT_STREAMS: list["TerminalTranscriptStream"] = []


def _is_terminal_debug_enabled() -> bool:
	try:
		return bool(get_config().get("terminal_debug", True))
	except Exception:
		return True


def _strip_ansi(text: str) -> str:
	return _ANSI_ESCAPE_RE.sub("", text)


class TerminalTranscriptStream:
	"""Tee stream that mirrors terminal output into terminal.txt with timestamps."""

	_session_header_written = False

	def __init__(self, stream: TextIO, transcript_path: Path) -> None:
		self._stream = stream
		self._transcript_path = transcript_path
		self._buffer = ""
		self._write_session_header()
		_TRANSCRIPT_STREAMS.append(self)

	def write(self, text: str) -> int:
		if not text:
			return 0

		self._stream.write(text)
		self._stream.flush()

		clean_text = _strip_ansi(text).replace("\r\n", "\n")
		if "\r" in clean_text and "\n" not in clean_text:
			return len(text)
		if "\r" in clean_text:
			clean_text = "\n".join(part.rsplit("\r", 1)[-1] for part in clean_text.split("\n"))

		with _TRANSCRIPT_LOCK:
			self._buffer += clean_text
			self._flush_complete_lines()

		return len(text)

	def flush(self) -> None:
		self._stream.flush()

	def finalize(self) -> None:
		with _TRANSCRIPT_LOCK:
			if self._buffer:
				self._write_transcript_line(self._buffer)
				self._buffer = ""

	def isatty(self) -> bool:
		return bool(getattr(self._stream, "isatty", lambda: False)())

	def fileno(self) -> int:
		return int(self._stream.fileno())

	@property
	def encoding(self) -> str:
		return getattr(self._stream, "encoding", "utf-8") or "utf-8"

	def __getattr__(self, item: str):
		return getattr(self._stream, item)

	def _write_session_header(self) -> None:
		with _TRANSCRIPT_LOCK:
			if self.__class__._session_header_written:
				return
			self._transcript_path.parent.mkdir(parents=True, exist_ok=True)
			with open(self._transcript_path, "a", encoding="utf-8") as handle:
				handle.write("\n")
				handle.write("=" * 88 + "\n")
				handle.write(
					f"[SESSION START] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
				)
				handle.write(f"[WORKDIR] {Path.cwd()}\n")
				handle.write(f"[TERMINAL FILE] {self._transcript_path}\n")
				handle.write("=" * 88 + "\n")
			self.__class__._session_header_written = True

	def _flush_complete_lines(self) -> None:
		while "\n" in self._buffer:
			line, self._buffer = self._buffer.split("\n", 1)
			self._write_transcript_line(line)

	def _write_transcript_line(self, line: str) -> None:
		clean_line = line.rstrip()
		with open(self._transcript_path, "a", encoding="utf-8") as handle:
			timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			if clean_line:
				handle.write(f"[{timestamp}] {clean_line}\n")
			else:
				handle.write("\n")


def _finalize_transcripts() -> None:
	for stream in list(_TRANSCRIPT_STREAMS):
		try:
			stream.finalize()
		except Exception:
			pass


if _is_terminal_debug_enabled():
	_stdout_transcript = TerminalTranscriptStream(sys.stdout, _TRANSCRIPT_PATH)
	_stderr_transcript = TerminalTranscriptStream(sys.stderr, _TRANSCRIPT_PATH)
	sys.stdout = _stdout_transcript
	sys.stderr = _stderr_transcript
	atexit.register(_finalize_transcripts)
	# Singleton console instance to be used throughout the application.
	# This ensures logging, progress bars, and interactive output share the same terminal state.
	console = Console(file=_stdout_transcript)
else:
	console = Console(file=sys.stdout)

