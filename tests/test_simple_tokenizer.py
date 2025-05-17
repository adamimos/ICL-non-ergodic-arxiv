import os
import sys
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from icl_non_ergodic_arxiv.simple_tokenizer import count_whitespace_tokens


class TestSimpleTokenizer(unittest.TestCase):
    def test_count_whitespace_tokens(self):
        self.assertEqual(count_whitespace_tokens("hello world"), 2)
        self.assertEqual(count_whitespace_tokens("  spaced   out text  "), 3)
        self.assertEqual(count_whitespace_tokens(""), 0)


if __name__ == "__main__":
    unittest.main()
