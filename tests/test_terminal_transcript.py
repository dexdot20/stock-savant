import re
import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from core.console import TerminalTranscriptStream


class TerminalTranscriptStreamTests(unittest.TestCase):
    def test_writes_timestamped_readable_lines(self) -> None:
        with TemporaryDirectory() as tmpdir:
            transcript_path = Path(tmpdir) / "terminal.txt"
            mirror = StringIO()
            previous_state = TerminalTranscriptStream._session_header_written
            TerminalTranscriptStream._session_header_written = False
            try:
                stream = TerminalTranscriptStream(mirror, transcript_path)

                stream.write("\x1b[31mHello\x1b[0m world\nNext line")
                stream.flush()
                stream.finalize()

                content = transcript_path.read_text(encoding="utf-8")

                self.assertIn("[SESSION START]", content)
                self.assertIn("Hello world", content)
                self.assertIn("Next line", content)
                self.assertNotIn("\x1b[31m", content)
                self.assertRegex(
                    content,
                    r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] Hello world",
                )
            finally:
                TerminalTranscriptStream._session_header_written = previous_state

    def test_flush_does_not_duplicate_partial_lines(self) -> None:
        with TemporaryDirectory() as tmpdir:
            transcript_path = Path(tmpdir) / "terminal.txt"
            mirror = StringIO()
            previous_state = TerminalTranscriptStream._session_header_written
            TerminalTranscriptStream._session_header_written = False
            try:
                stream = TerminalTranscriptStream(mirror, transcript_path)

                stream.write("Partial line")
                stream.flush()

                content = transcript_path.read_text(encoding="utf-8")
                self.assertIn("[SESSION START]", content)
                self.assertNotIn("Partial line", content)

                stream.write(" completed\n")
                stream.finalize()

                content = transcript_path.read_text(encoding="utf-8")
                self.assertIn("Partial line completed", content)
                self.assertEqual(content.count("Partial line completed"), 1)
            finally:
                TerminalTranscriptStream._session_header_written = previous_state


if __name__ == "__main__":
    unittest.main()
