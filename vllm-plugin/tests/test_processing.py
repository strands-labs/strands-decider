"""Image preprocessing for the multimodal architectures: placeholder counts for images the server has resized, the PIL
image processor, and their registration on the plugin's classes only. Needs vLLM (no GPU)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

processing = pytest.importorskip("strands_decider_vllm.processing")
model = pytest.importorskip("strands_decider_vllm.model")


@pytest.mark.parametrize(
    "w, h, n",
    [(48, 48, 1), (912, 672, 266), (2160, 288, 270), (768, 768, 256), (48, 13440, 280)],
)
def test_a_resized_image_has_one_token_per_48_pixels(w, h, n):
    assert processing.soft_tokens_of_resized(w, h, 48, 280) == n


@pytest.mark.parametrize("w, h", [(224, 224), (912, 671), (47, 48)])
def test_an_image_that_was_not_resized_is_refused(w, h):
    with pytest.raises(ValueError, match="multiples of 48"):
        processing.soft_tokens_of_resized(w, h, 48, 280)


def test_more_than_the_processors_tokens_is_refused():
    with pytest.raises(ValueError, match="not 1 to 280"):
        processing.soft_tokens_of_resized(48 * 17, 48 * 17, 48, 280)


def test_the_spec_resize_is_not_a_fixed_point_of_vllms_formula():
    """Why the plugin counts placeholders itself: vLLM's formula, applied to an image already at W x H, is not N."""
    from strands_decider_vllm.server import resized_size

    def vllm_n(w, h):  # Gemma4ProcessingInfo._compute_num_soft_tokens, max_soft_tokens 280
        import math

        scale = math.sqrt(280 * 9 / ((h / 16) * (w / 16)))
        th = max(48, math.floor(h * scale / 48) * 48)
        tw = max(48, math.floor(w * scale / 48) * 48)
        return min((th // 16) * (tw // 16) // 9, 280)

    w, h = resized_size(1, 2)  # 528 x 1104: 11 x 23 = 253 tokens of 48 pixels
    assert processing.soft_tokens_of_resized(w, h, 48, 280) == 253 and vllm_n(w, h) == 264


def test_the_plugins_classes_carry_the_plugins_processor_and_vllms_keep_theirs():
    from vllm.model_executor.models.gemma4_mm import (
        Gemma4ForConditionalGeneration,
        Gemma4MultiModalProcessor,
        Gemma4ProcessingInfo,
    )
    from vllm.model_executor.models.gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration,
        Gemma4UnifiedProcessingInfo,
    )

    mm = model.StrandsDeciderForConditionalGeneration._processor_factory
    assert (
        mm.processor is processing.DeciderMultiModalProcessor
        and mm.info is processing.DeciderProcessingInfo
    )
    uni = model.StrandsDeciderUnifiedForConditionalGeneration._processor_factory
    assert (
        uni.processor is processing.DeciderMultiModalProcessor
        and uni.info is Gemma4UnifiedProcessingInfo
    )
    assert Gemma4ForConditionalGeneration._processor_factory.processor is Gemma4MultiModalProcessor
    assert Gemma4ForConditionalGeneration._processor_factory.info is Gemma4ProcessingInfo
    assert (
        Gemma4UnifiedForConditionalGeneration._processor_factory.info is Gemma4UnifiedProcessingInfo
    )
    assert issubclass(processing.DeciderMultiModalProcessor, Gemma4MultiModalProcessor)


def test_pin_pil_swaps_the_image_processor_once(monkeypatch):
    transformers = pytest.importorskip("transformers")
    loads = []

    class FakePil(transformers.Gemma4ImageProcessorPil):
        def __init__(self):  # no files needed
            pass

    def from_pretrained(model_or_path, revision=None):
        loads.append((model_or_path, revision))
        return FakePil()

    monkeypatch.setattr(
        transformers.Gemma4ImageProcessorPil, "from_pretrained", staticmethod(from_pretrained)
    )
    proc = SimpleNamespace(image_processor=object())
    mc = SimpleNamespace(model="/ckpt", revision="r")
    processing.pin_pil(proc, mc)
    assert isinstance(proc.image_processor, transformers.Gemma4ImageProcessorPil)
    processing.pin_pil(proc, mc)
    assert loads == [("/ckpt", "r")], "loaded once, then left as it is"
