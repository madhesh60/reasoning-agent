import asyncio
import unittest
from unittest.mock import MagicMock

from logos.cli import DynamicStatus


class TestDynamicStatus(unittest.IsolatedAsyncioTestCase):
    async def test_dynamic_status_cycles_phrases(self):
        console_mock = MagicMock()
        status_mock = MagicMock()
        console_mock.status.return_value = status_mock

        phrases = ["Phrase A", "Phrase B"]

        async with DynamicStatus(console_mock, "Initial", phrases, spinner_style="blue"):
            await asyncio.sleep(4.5)  # enough to trigger at least one update loop (which has a 4s sleep)

        status_mock.update.assert_called()
        call_args = status_mock.update.call_args[0][0]
        self.assertIn("Phrase A", call_args)
        self.assertIn("blue", call_args)

if __name__ == "__main__":
    unittest.main()
