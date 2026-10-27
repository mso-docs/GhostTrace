"""Model unloading must release child-module references, not just the root model."""
import gc
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
from ghosttrace.models.qwen import QwenAdapter


class ModuleGroup:pass
class RootModel:pass


class AdapterLifecycle(unittest.TestCase):
    def test_unload_releases_root_and_block_module_references(self):
        adapter=QwenAdapter();model=RootModel();model.layers=ModuleGroup()
        root_reference=weakref.ref(model);block_reference=weakref.ref(model.layers)
        adapter.model=model;adapter.blocks=model.layers;adapter.tokenizer=object()
        del model
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
        with patch.dict('sys.modules',{'torch':fake_torch}):adapter.unload()
        gc.collect()
        self.assertIsNone(root_reference())
        self.assertIsNone(block_reference(),'A retained block list keeps almost all model weights resident after unload.')
        self.assertIsNone(adapter.tokenizer)
