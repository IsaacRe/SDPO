# Layerwise Distillation Plan for SDPO

## Background

This note captures a concrete plan for adding a layerwise distillation loss to the SDPO codebase.

The starting point is:

- SDPO's existing on-policy self-distillation implementation in this repo
- NVIDIA ModelOpt's layerwise distillation implementation in `modelopt/torch/distill/layerwise_distillation_model.py`
- The SDPO paper: https://arxiv.org/abs/2601.20802
- The NVIDIA layerwise distillation reference paper: https://arxiv.org/abs/2407.14679

The relevant local repos are:

- `SDPO`: `/home/isaac/SDPO`
- `Model-Optimizer`: `/home/isaac/Model-Optimizer`

## What SDPO Already Does

SDPO already has a self-distillation path in the actor update:

- The trainer constructs reprompted teacher inputs in `verl/trainer/ppo/ray_trainer.py`
- The actor update runs a student forward and a teacher forward in `verl/workers/actor/dp_actor.py`
- The two outputs are combined with the current logits-based self-distillation objective in `verl/trainer/ppo/core_algos.py`

The key code paths are:

- `verl/workers/actor/dp_actor.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/workers/fsdp_workers.py`
- `verl/workers/config/actor.py`
- `verl/trainer/config/actor/actor.yaml`

Current behavior:

- SDPO supports full-logit KD or top-k logit KD
- The teacher is either the ref model or a trust-region mixed teacher
- Distillation is only active when `policy_loss.loss_mode == "sdpo"`
- The teacher input is not identical to the student input: it is a reprompted input that may include a prior successful attempt and/or environment feedback

This last point is the main design constraint for layerwise KD in SDPO.

## What ModelOpt Does

ModelOpt's distillation code provides two useful patterns:

1. Hook-based activation capture
   - `DistillationModel` maps `(student_layer_name, teacher_layer_name)` pairs to loss functions
   - It captures intermediate outputs from student and teacher layers with forward hooks
   - It computes KD losses from those cached activations

2. Layerwise mode with teacher-input bypass
   - `LayerwiseDistillationModel` can inject teacher layer inputs into corresponding student layers
   - It freezes all student layers except the targeted layers
   - It can disable LM heads to save compute

The main relevant files are:

- `/home/isaac/Model-Optimizer/modelopt/torch/distill/distillation_model.py`
- `/home/isaac/Model-Optimizer/modelopt/torch/distill/layerwise_distillation_model.py`

## Important Difference Between SDPO and ModelOpt

ModelOpt's layerwise design is naturally suited to settings where student and teacher are processing the same input stream, or where the student is a partially replaced version of the teacher.

SDPO is different:

- Student forward uses the original prompt
- Teacher forward uses a reprompted prompt
- The generated response tokens are shared, but the prefix context can differ substantially

Because of that, a direct transplant of ModelOpt's teacher-input bypass approach is probably not the right first implementation for SDPO.

The safest first version is to:

- keep SDPO's existing separate student and teacher forwards
- capture selected hidden states with hooks
- compare only the response-token activations
- combine the resulting layerwise loss with the existing logits KD loss

## Recommended First Implementation

### 1. Extend self-distillation config

Add layerwise KD options to `SelfDistillationConfig` in `verl/workers/config/actor.py` and surface them in `verl/trainer/config/actor/actor.yaml`.

Suggested fields:

- `layerwise_enabled: bool = False`
- `layer_pairs: list[str] | list[tuple[str, str]]`
- `layer_loss_type: str = "mse"`
- `layerwise_weight: float = 1.0`
- `layerwise_mask_scope: str = "response_only"`
- `layerwise_combine_mode: str = "add"`
- `layer_norm_targets: bool = False`
- `detach_teacher_activations: bool = True`

Possible future fields:

- `layerwise_freeze_non_target_layers: bool = False`
- `teacher_input_bypass: bool = False`

The future fields should not be enabled in the first version.

### 2. Add an activation capture utility

Implement a small local hook manager in SDPO rather than importing ModelOpt's meta-model wrapper directly.

Responsibilities:

- resolve named submodules on student and teacher
- register forward hooks on selected layer pairs
- cache intermediate outputs for a single microbatch
- clear cached activations after loss computation
- safely remove handles at teardown

Why this approach:

- SDPO already owns the training loop and explicitly runs both student and teacher forwards
- a lightweight hook utility is much easier to integrate than wrapping the actor module in a new distillation meta-model
- this avoids invasive changes to FSDP integration

### 3. Capture activations during the existing SDPO actor update

The integration point should stay in `verl/workers/actor/dp_actor.py`, where SDPO already does:

- student forward
- teacher forward
- self-distillation loss computation

The intended flow is:

1. If `layerwise_enabled`, enable hook capture before the student forward
2. Run student forward as today
3. Run teacher forward under `torch.no_grad()` as today
4. Compute logits KD loss as today
5. Compute layerwise KD loss from cached activations
6. Combine the two losses
7. Log separate metrics for logits KD and layerwise KD
8. Clear cached activations

This keeps the current SDPO logic intact and makes the layerwise term an additive extension.

### 4. Distill only response-token activations in v1

This is the critical design choice.

Because student and teacher prefixes differ, full-sequence hidden-state matching is likely to be noisy or actively misleading. The first implementation should align only the response positions already used in SDPO's current token-level KD.

Practical implication:

- Hook outputs will likely be `[batch, seq, hidden]`
- The loss should slice the response region only
- The same `response_mask` and `self_distillation_mask` logic used by current SDPO KD should be reused

This makes the layerwise objective compatible with SDPO's reprompt-based teacher.

### 5. Implement layerwise loss separately from existing logits KD

Do not overload the current `compute_self_distillation_loss` too aggressively on day one.

Preferred structure:

- keep `compute_self_distillation_loss(...)` for the current logits KD
- add a new helper such as `compute_layerwise_self_distillation_loss(...)`
- combine the two in `dp_actor.py`

Benefits:

- easier ablations
- simpler metrics
- lower regression risk
- easier to disable one loss without affecting the other

Suggested metrics:

- `self_distillation/logits_kd_loss`
- `self_distillation/layerwise_kd_loss`
- `self_distillation/total_kd_loss`
- `self_distillation/num_layer_pairs`
- per-layer losses if not too expensive to log

### 6. Start with a simple loss family

Recommended v1 options:

- MSE on hidden states
- Smooth L1 as optional alternative
- Cosine distance only if needed later

For the first pass, MSE is the safest baseline.

Masking/reduction:

- compute tokenwise loss over hidden dimension
- reduce hidden dimension first
- apply response mask and self-distillation mask
- aggregate with the same `agg_loss` helper already used by SDPO

### 7. Keep teacher wiring unchanged initially

The current teacher construction in `verl/workers/fsdp_workers.py` should remain as-is for the first implementation:

- `ema` teacher -> use ref model
- `trust-region` teacher -> use mixed teacher wrapper

This allows layerwise KD to immediately piggyback on SDPO's current teacher semantics.

Potential caveat:

- if the trust-region teacher wrapper changes module names or wrapping depth, hook registration may need special handling

That should be validated explicitly.

## Why Not Use ModelOpt's LayerwiseDistillationModel Directly

Using ModelOpt directly is tempting, but there are several reasons not to start there:

1. SDPO already has an explicit two-forward training loop.
   ModelOpt's wrapper is designed to encapsulate teacher and student inside one meta-model, which is unnecessary here.

2. ModelOpt's layerwise mode assumes a much tighter teacher/student correspondence.
   In SDPO the teacher sees a reprompted input, so same-layer same-position matching is not generally valid outside the response region.

3. ModelOpt's layerwise mode freezes most of the student by default.
   That is not obviously compatible with PPO-style policy optimization in SDPO.

4. SDPO's FSDP and actor/ref worker setup is already specialized.
   A small hook-based extension is lower-risk than introducing a new top-level wrapper abstraction.

ModelOpt should be treated as a design reference for:

- layer-pair configuration
- hook registration
- cached activation loss computation

Not as a drop-in implementation.

## Proposed Implementation Order

### Phase 1: plumbing

- add config fields
- add validation
- add layer-pair parsing and module resolution
- add hook manager utility

### Phase 2: loss path

- add layerwise activation capture in the actor update
- add response-only layerwise KD loss computation
- combine with existing logits KD loss
- add metrics

### Phase 3: verification

- run a tiny SDPO training step with `layerwise_enabled=False` to confirm no regression
- run a tiny step with `layerwise_enabled=True` and one late transformer block
- check hook cleanup, memory use, and shape alignment
- verify behavior under both full-logit KD and top-k KD

### Phase 4: hardening

- validate FSDP module naming
- validate `use_remove_padding`
- validate trust-region teacher path
- validate failure modes when a requested layer name is missing

## Main Risks

### 1. Hidden-state alignment may be semantically weak

Because the teacher prefix is reprompted, the teacher and student hidden states may differ for reasons unrelated to response quality. Restricting to response tokens reduces this risk but does not remove it.

### 2. Hooked activations may be expensive

Capturing multiple deep layers for both student and teacher can increase memory pressure. Start with a small number of layer pairs.

### 3. FSDP-wrapped module names may differ from raw HF module names

The layer-pair config should be validated against the actual wrapped actor and teacher modules used in training.

### 4. `use_remove_padding` may complicate layer output shapes

The actor forward path supports unpadding and sequence-parallel behavior. Hidden-state extraction must be checked carefully to ensure the captured tensors correspond to padded sequence positions expected by the masking logic.

## Suggested Initial Experiment

For the first experiment:

- keep current SDPO logits KD enabled
- enable layerwise KD on 1 to 3 late transformer blocks
- use MSE
- distill response tokens only
- use a small `layerwise_weight`
- log the two KD losses separately

This gives the cleanest signal about whether layerwise information helps beyond current logits KD.

## Concrete Files Likely To Change

- `/home/isaac/SDPO/verl/workers/config/actor.py`
- `/home/isaac/SDPO/verl/trainer/config/actor/actor.yaml`
- `/home/isaac/SDPO/verl/workers/actor/dp_actor.py`
- `/home/isaac/SDPO/verl/trainer/ppo/core_algos.py`

Possible new file:

- a small hook/capture utility under `verl/workers/actor/` or `verl/utils/`

## Bottom Line

The best first implementation is not a full ModelOpt port.

It is a targeted SDPO-native extension:

- add configurable layer-pair hooks
- capture student and teacher activations during the existing actor update
- compute a response-only layerwise KD loss
- add it to the current SDPO logits KD objective

If that works, then more aggressive variants such as teacher-input bypass or partial layer freezing can be explored later as explicit experiments rather than being built into the first version.
