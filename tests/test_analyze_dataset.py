import sys
import types
import importlib
import argparse
import unittest

class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

class DummyDataset(list):
    def __init__(self, rows, label_names):
        super().__init__(rows)
        self.features = {"label": types.SimpleNamespace(names=label_names)}

class AnalyzeDatasetTests(unittest.TestCase):
    def setUp(self):
        datasets_module = types.ModuleType("datasets")

        def load_dataset(name, config, split):
            return DummyDataset(
                [
                    {"text": "a b c", "label": 0},
                    {"text": "d e", "label": 1},
                    {"text": "", "label": None},
                ],
                ["catA", "catB"],
            )

        datasets_module.load_dataset = load_dataset
        transformers_module = types.ModuleType("transformers")
        transformers_module.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda name: DummyTokenizer()
        )
        self.patchers = []
        self._patch(sys.modules, "datasets", datasets_module)
        self._patch(sys.modules, "transformers", transformers_module)

    def tearDown(self):
        for target, name in self.patchers:
            if name in target:
                del target[name]

    def _patch(self, target, name, value):
        target[name] = value
        self.patchers.append((target, name))

    def test_parse_args_defaults(self):
        sys.argv = ["prog"]
        import scripts.analyze_dataset as analyze_dataset
        args = analyze_dataset.parse_args()
        self.assertEqual(args.top_n, 11)
        self.assertEqual(args.tokenizer, "bert-base-uncased")
        self.assertEqual(args.split, "train")

    def test_main_output(self):
        module = importlib.import_module("scripts.analyze_dataset")
        original = module.parse_args
        module.parse_args = lambda: argparse.Namespace(top_n=11, tokenizer="dummy", split="train")
        from io import StringIO
        import contextlib
        output = StringIO()
        with contextlib.redirect_stdout(output):
            module.main()
        module.parse_args = original
        out = output.getvalue()
        self.assertIn("catA\t3", out)
        self.assertIn("catB\t2", out)

if __name__ == "__main__":
    unittest.main()
