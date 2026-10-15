"""Keep the pinned upstream penalty kernel independent of ImmersX fixes."""
import hashlib
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[3]


class CouplingBackport(unittest.TestCase):
    def test_penalty_kernel_matches_pinned_step80_revision(self):
        text = (ROOT / 'include/immersx/compatibility/dealii_9_9/dof_handler_coupling.h').read_text()
        start = text.index('{', text.index('create_nitsche_restriction_matrix('))
        level = 0
        for end in range(start, len(text)):
            if text[end] == '{':
                level += 1
            elif text[end] == '}':
                level -= 1
            if not level:
                break
        body = re.sub(r'\s+', '', text[start:end + 1])
        # Computed directly from deal.II PR #18888 at c061c24bd41b4eb9fb62d1625dc0b818a54bc2bd.
        self.assertEqual(hashlib.sha256(body.encode()).hexdigest(),
                         'ce3b572d8c2804402205e0d8d98edd5d947f59aea2e00344e073cb440b2001b5')
        self.assertIn('SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later', text)


if __name__ == '__main__':
    unittest.main()
