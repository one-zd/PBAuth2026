# Licensed under the Apache License, Version 2.0 (the "License");
# (保留/遵守原始 license)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel

# from ...utils import logging

# logger = logging.get_logger(__name__)


def cosine_distance(image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
    """
    image_embeds: (B, D)
    text_embeds: (N, D)
    return: (B, N) matrix of cosine similarities (torch)
    """
    # normalize along dim=1
    normalized_image_embeds = F.normalize(image_embeds, p=2, dim=-1)
    normalized_text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    # (B, D) @ (D, N) -> (B, N)
    return torch.matmul(normalized_image_embeds, normalized_text_embeds.t())


class PrivateDifferentiableSafetyChecker(PreTrainedModel):
    """
    A modified, differentiable variant of HuggingFace's StableDiffusionSafetyChecker.
    - All computations use torch tensors (no .cpu().numpy()), so outputs are differentiable
      w.r.t. the provided clip_input / pixel_values (if they require_grad).
    - Returns (maybe_blackened_images, nsfw_logits, nsfw_probs, has_nsfw_mask)
      where nsfw_logits is a differentiable scalar per image that you can use with
      BCEWithLogitsLoss to supervise upstream models.
    """

    config_class = CLIPConfig
    main_input_name = "clip_input"
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        # keep same architecture pieces as official
        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        # concept embeddings (17 concepts) and special care embeds (3) -- keep them as buffers/params
        # by default keep requires_grad=False so we don't train them unless you explicitly allow
        self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        self.concept_embeds_weights = nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = nn.Parameter(torch.ones(3), requires_grad=False)

    def forward(
        self,
        clip_input: torch.Tensor,  # pixel_values (B, C, H, W) or preprocessed inputs suitable for CLIPVisionModel
        images: torch.Tensor = None,  # optional raw images (B,C,H,W) if you want blackening behavior
        adjustment: float = 0.0,
        blacken_images: bool = True,
        nsfw_threshold: float = 0.0,
    ):
        """
        Args:
            clip_input: tensor acceptable by CLIPVisionModel (usually pixel_values), shape (B, C, H, W).
                        IMPORTANT: to get gradients back to your generated image, make sure this tensor has requires_grad=True.
            images: optional same-shape images to be potentially blackened (if blacken_images True).
                    If None, we will operate on clip_input for blackening.
            adjustment: float, optional extra margin to make filter stronger/slacker (default 0.0)
            blacken_images: if True, set nsfw images to black (mimic official behavior)
            nsfw_threshold: decision threshold on logits (default 0.0). Optionally ignored if you prefer using probs.

        Returns:
            (out_images, nsfw_logits, nsfw_probs, has_nsfw_mask)
            - out_images: tensor same as images or clip_input, with blackened entries if any and blacken_images True.
            - nsfw_logits: tensor (B,) logits representing NSFW score (higher -> more NSFW). These are differentiable.
            - nsfw_probs: torch.sigmoid(nsfw_logits) (B,)
            - has_nsfw_mask: boolean tensor (B,) indicating nsfw_logits > nsfw_threshold (non-differentiable mask)
        """

        # 1) get pooled output from vision_model (this is differentiable w.r.t. clip_input)
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output shape (B, hidden)
        image_embeds = self.visual_projection(pooled_output)  # (B, proj_dim)

        # 2) compute cosine distances (torch tensors)
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)  # (B, 3)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)  # (B, 17)

        # 3) compute scores in a vectorized, differentiable way
        # special adjustment: if any special score > weight, increase adjustment a bit
        special_scores = special_cos_dist - self.special_care_embeds_weights  # (B,3)
        # special care boolean (B,) but still torch boolean (non-diff) for decision-making
        special_care_mask = special_scores > 0  # (B,3) boolean

        # special adjustment per sample: if any special True -> +0.01 (as in official)
        # Use differentiable formulation: special_adjustment_scalar is float tensor depending on special_scores (non-grad).
        # We keep it non-differentiable (it's just a discrete tweak), but you can remove this if you want pure differentiability.
        special_adjustment = (special_care_mask.any(dim=1).float() * 0.01).unsqueeze(1)  # (B,1)

        # concept scores: subtract concept weights and add special adjustment
        concept_scores = cos_dist - self.concept_embeds_weights.unsqueeze(0) + special_adjustment  # (B,17)

        # We want a single per-image NSFW logit. A reasonable differentiable choice:
        # take the maximum over concept_scores (max margin). max is piecewise-differentiable (subgradient),
        # PyTorch provides autograd for max (grad flows to argmax index).
        nsfw_logits_vals, _ = concept_scores.max(dim=1)  # (B,)

        # Optionally allow further adjustment scalar
        nsfw_logits = nsfw_logits_vals + adjustment  # (B,)

        # convert to probability
        nsfw_probs = torch.sigmoid(nsfw_logits)

        # create non-differentiable boolean mask for compatibility with original API
        has_nsfw_concepts = nsfw_logits > nsfw_threshold  # boolean tensor (B,)

        # blacken images if requested (mimic official behavior)
        # we operate on a copy to avoid in-place on user tensors
        if images is None:
            out_images = clip_input.clone()
        else:
            out_images = images.clone()

        if blacken_images:
            # set nsfw indices to zeros (black); this operation is non-differentiable where mask applies (because it's assignment),
            # but nsfw_logits is still differentiable wrt clip_input.
            out_images = out_images * (~has_nsfw_concepts).view(-1, 1, 1, 1).to(out_images.dtype)

        # if has_nsfw_concepts.any():
        #     print(
        #         "Potential NSFW content detected in one or more images. "
        #         "Returned images may be blackened. Use nsfw_logits / nsfw_probs for differentiable supervision."
        #     )

        return out_images, nsfw_logits, nsfw_probs, has_nsfw_concepts

    def forward_onnx(self, clip_input: torch.Tensor, images: torch.Tensor):
        # A similar implementation suitable for ONNX export (keeps differentiable ops)
        pooled_output = self.vision_model(clip_input)[1]
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        special_scores = special_cos_dist - self.special_care_embeds_weights.unsqueeze(0)
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care.float().unsqueeze(1) * 0.01

        concept_scores = (cos_dist - self.concept_embeds_weights.unsqueeze(0)) + special_adjustment
        nsfw_logits, _ = concept_scores.max(dim=1)
        nsfw_probs = torch.sigmoid(nsfw_logits)

        images[nsfw_probs > 0.5] = 0.0
        has_nsfw_concepts = nsfw_probs > 0.5

        return images, nsfw_logits, nsfw_probs, has_nsfw_concepts
