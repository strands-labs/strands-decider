"""Image preprocessing for the plugin's multimodal architectures, so that vLLM encodes the core's
pixels into the core's number of placeholders.

The plugin's server resizes each image to the core's W x H (``resize_image``) and turns the image
processor's own resize off (``do_resize=False``). vLLM 0.33 would not follow it in two places:

- vLLM counts an image's placeholders by its own formula, which assumes the processor resizes: on an
  image already at W x H it misses the core's count for 1,827 of 4,906 sizes. The plugin counts one
  per 48 x 48 pixels (patch size times pooling kernel), what the processor gives with its resize off.
  vLLM's startup profiling keeps the resize on, and vLLM's count.
- transformers 5.20 resolves ``Gemma4ImageProcessor`` to its torchvision backend, which converts
  pixels to floating point differently from ``Gemma4ImageProcessorPil``, the reference's for E2B,
  E4B, 26B-A4B and 31B. The plugin loads the PIL one. (``--mm-processor-kwargs '{"backend": "pil"}'``
  can't select it: the video processor, which has no PIL backend, gets it too.) The 12B's
  ``Gemma4UnifiedImageProcessor`` has only a torchvision backend, which the reference uses as well.

Both are registered for the plugin's classes only (``model``); vLLM's Gemma 4 classes keep theirs.
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


class PilProcessingInfo(Gemma4ProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> Any:
        from transformers import Gemma4ImageProcessorPil

        processor = super().get_hf_processor(**kwargs)
        if not isinstance(processor.image_processor, Gemma4ImageProcessorPil):  # once per processor
            mc = self.ctx.model_config
            processor.image_processor = Gemma4ImageProcessorPil.from_pretrained(
                mc.model, revision=mc.revision
            )
        return processor


class ResizedImageProcessor(Gemma4MultiModalProcessor):
    def _get_prompt_updates(
        self, mm_items: Any, hf_processor_mm_kwargs: Mapping[str, Any], out_mm_kwargs: Any
    ) -> Sequence[PromptUpdate]:
        updates = super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
        resize = self.info.ctx.get_merged_mm_kwargs(hf_processor_mm_kwargs).get("do_resize", True)
        if "image" not in mm_items or resize is not False:
            return updates
        config = self.info.get_hf_config()
        unit = config.vision_config.patch_size * config.vision_config.pooling_kernel_size
        image_id = self.info.get_hf_processor(**hf_processor_mm_kwargs).image_token_id
        images = mm_items.get_items("image", ImageProcessorItems)

        def replacement(item_idx: int) -> PromptUpdateDetails:
            size = images.get_image_size(item_idx)
            n = (size.width // unit) * (size.height // unit)
            ids = [config.boi_token_id, *[image_id] * n, config.eoi_token_id]
            return PromptUpdateDetails.select_token_id(ids, image_id)

        image = PromptReplacement(modality="image", target=[image_id], replacement=replacement)
        return [image if u.modality == "image" else u for u in updates]


register_multimodal = MULTIMODAL_REGISTRY.register_processor(
    ResizedImageProcessor, info=PilProcessingInfo, dummy_inputs=Gemma4DummyInputsBuilder
)
register_unified = MULTIMODAL_REGISTRY.register_processor(
    ResizedImageProcessor, info=Gemma4UnifiedProcessingInfo, dummy_inputs=Gemma4DummyInputsBuilder
)
