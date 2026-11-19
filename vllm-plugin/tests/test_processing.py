"""The plugin's image processing is registered for its own classes only, and loads the PIL image
processor once. Needs vLLM, no GPU; the GPU tiers check the pixels and placeholder counts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

processing = pytest.importorskip("strands_decider_vllm.processing")
model = pytest.importorskip("strands_decider_vllm.model")


def test_the_plugins_classes_carry_its_processing_and_vllms_keep_theirs():
    from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration
    from vllm.model_executor.models.gemma4_unified import Gemma4UnifiedForConditionalGeneration

    for cls, info in [
        (model.StrandsDeciderForConditionalGeneration, processing.PilProcessingInfo),
        (
            model.StrandsDeciderUnifiedForConditionalGeneration,
            processing.Gemma4UnifiedProcessingInfo,
        ),
    ]:
        assert cls._processor_factory.processor is processing.ResizedImageProcessor
        assert cls._processor_factory.info is info
    assert (
        Gemma4ForConditionalGeneration._processor_factory.processor
        is processing.Gemma4MultiModalProcessor
    )
    assert Gemma4ForConditionalGeneration._processor_factory.info is processing.Gemma4ProcessingInfo
    assert (
        Gemma4UnifiedForConditionalGeneration._processor_factory.processor
        is not processing.ResizedImageProcessor
    )


def test_the_pil_image_processor_is_loaded_once(monkeypatch):
    transformers = pytest.importorskip("transformers")
    loads = []

    class Pil(transformers.Gemma4ImageProcessorPil):
        def __init__(self):  # no files needed
            pass

    monkeypatch.setattr(
        transformers.Gemma4ImageProcessorPil,
        "from_pretrained",
        staticmethod(lambda model, revision=None: loads.append((model, revision)) or Pil()),
    )
    hf = SimpleNamespace(image_processor=object())
    monkeypatch.setattr(processing.Gemma4ProcessingInfo, "get_hf_processor", lambda self, **kw: hf)
    info = object.__new__(processing.PilProcessingInfo)
    info.ctx = SimpleNamespace(model_config=SimpleNamespace(model="/ckpt", revision="r"))
    for _ in range(2):
        assert isinstance(
            info.get_hf_processor().image_processor, transformers.Gemma4ImageProcessorPil
        )
    assert loads == [("/ckpt", "r")]
