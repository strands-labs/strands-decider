"""Image preprocessing for the plugin's multimodal architectures: the specification's pixels, and placeholder counts
that match them.

The specification (section 11.1) resizes a decoded image to its W x H with Pillow's bicubic resampling, then runs the
checkpoint's image processor with its own resize off (``do_resize=False``). The plugin's server does the resize and
passes ``do_resize=False`` with each image request. Two things in vLLM 0.33 would not follow:

- **The placeholder count.** vLLM expands each ``<|image|>`` into N placeholders by its own formula on the image's
  size (``Gemma4ProcessingInfo._compute_num_soft_tokens``), which assumes the processor will resize. On an image
  already at W x H that formula is not N for 1,787 of the 3,200 sizes of the core's soft-token table, and the encoder's
  output would then not fit the placeholders. :class:`DeciderMultiModalProcessor` counts one placeholder per 48 x 48
  pixels (patch size times pooling kernel) when the request turns the resize off, which is what the processor
  produces, and refuses an image whose sides are not multiples of 48. With the resize on (vLLM's own startup
  profiling), vLLM's count is kept.
- **The image processor's backend.** ``processor_config.json`` names ``Gemma4ImageProcessor``, which transformers 5.20
  resolves to its torchvision backend whenever torchvision is installed. With the resize off it still converts the
  pixels to floating point differently from ``Gemma4ImageProcessorPil``, the backend the specification's reference uses
  for E2B, E4B, 26B-A4B and 31B. :class:`DeciderProcessingInfo` gives vLLM's processor the PIL image processor, loaded
  from the same checkpoint. The 12B's ``Gemma4UnifiedImageProcessor`` has a torchvision backend only, which the
  reference uses too. (vLLM's ``--mm-processor-kwargs '{"backend": "pil"}'`` cannot select it: the processor passes the
  backend to its video processor as well, which has no PIL backend.)

Both are registered for the plugin's own classes only (:mod:`strands_decider_vllm.model`), as vLLM registers a model's
processor; vLLM's Gemma 4 classes are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vllm.model_executor.models.gemma4_mm import (
    Gemma4DummyInputsBuilder,
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
)
from vllm.model_executor.models.gemma4_unified import Gemma4UnifiedProcessingInfo
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageProcessorItems
from vllm.multimodal.processing import PromptReplacement, PromptUpdate, PromptUpdateDetails

from . import logger


def pin_pil(processor: Any, model_config: Any) -> Any:
    """``processor`` with its image processor on the PIL backend (idempotent)."""
    from transformers import Gemma4ImageProcessorPil

    if isinstance(processor.image_processor, Gemma4ImageProcessorPil):
        return processor
    pil = Gemma4ImageProcessorPil.from_pretrained(
        model_config.model, revision=model_config.revision
    )
    logger.info(
        "strands-decider: preparing images with %s (the specification's reference) in place of %s",
        type(pil).__name__,
        type(processor.image_processor).__name__,
    )
    processor.image_processor = pil
    return processor


class DeciderProcessingInfo(Gemma4ProcessingInfo):
    """Gemma 4's processing info with the PIL image processor (E2B, E4B, 26B-A4B, 31B)."""

    def get_hf_processor(self, **kwargs: object) -> Any:
        return pin_pil(super().get_hf_processor(**kwargs), self.ctx.model_config)


def soft_tokens_of_resized(width: int, height: int, unit: int, max_soft_tokens: int) -> int:
    """The soft tokens of an image already resized to W x H: one per ``unit`` x ``unit`` pixels."""
    if width % unit or height % unit:
        raise ValueError(
            f"strands-decider: an image sent with do_resize=False must be resized to the specification's W x H, "
            f"whose sides are multiples of {unit} pixels; got {width} x {height}"
        )
    n = (width // unit) * (height // unit)
    if not 1 <= n <= max_soft_tokens:
        raise ValueError(
            f"strands-decider: a {width} x {height} image has {n} soft tokens, not 1 to {max_soft_tokens}"
        )
    return n


class DeciderMultiModalProcessor(Gemma4MultiModalProcessor):
    """Gemma 4's multimodal processor, with placeholder counts for images the server has already resized."""

    def _get_prompt_updates(
        self,
        mm_items: Any,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: Any,
    ) -> Sequence[PromptUpdate]:
        updates = list(super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs))
        merged = self.info.ctx.get_merged_mm_kwargs(hf_processor_mm_kwargs)
        if "image" not in mm_items or merged.get("do_resize", True) is not False:
            return updates
        config = self.info.get_hf_config()
        vision = config.vision_config
        unit = vision.patch_size * vision.pooling_kernel_size
        max_soft = int(getattr(vision, "default_output_length", None) or vision.num_soft_tokens)
        hf = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        images = mm_items.get_items("image", ImageProcessorItems)

        def replacement(item_idx: int) -> PromptUpdateDetails:
            size = images.get_image_size(item_idx)
            n = soft_tokens_of_resized(size.width, size.height, unit, max_soft)
            ids = [config.boi_token_id] + [hf.image_token_id] * n + [config.eoi_token_id]
            return PromptUpdateDetails.select_token_id(ids, hf.image_token_id)

        return [
            PromptReplacement(modality="image", target=[hf.image_token_id], replacement=replacement)
            if u.modality == "image"
            else u
            for u in updates
        ]


#: Registered on StrandsDeciderForConditionalGeneration (E2B, E4B, 26B-A4B, 31B).
register_multimodal = MULTIMODAL_REGISTRY.register_processor(
    DeciderMultiModalProcessor, info=DeciderProcessingInfo, dummy_inputs=Gemma4DummyInputsBuilder
)
#: Registered on StrandsDeciderUnifiedForConditionalGeneration (12B).
register_unified = MULTIMODAL_REGISTRY.register_processor(
    DeciderMultiModalProcessor,
    info=Gemma4UnifiedProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
