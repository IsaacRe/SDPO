# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def _extract_activation_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Unsupported hooked activation output type: {type(output)}")


def _resolve_submodule(module: nn.Module, name: str) -> nn.Module:
    try:
        return module.get_submodule(name)
    except AttributeError:
        named_modules = dict(module.named_modules())
        if name in named_modules:
            return named_modules[name]
    except Exception:
        named_modules = dict(module.named_modules())
        if name in named_modules:
            return named_modules[name]
    raise ValueError(f"Could not resolve submodule '{name}' on module of type {type(module).__name__}")


def parse_layer_pairs(layer_pairs: Sequence[Any]) -> list[tuple[str, str]]:
    parsed_pairs: list[tuple[str, str]] = []
    for layer_pair in layer_pairs:
        if isinstance(layer_pair, str):
            if "=" in layer_pair:
                student_layer, teacher_layer = layer_pair.split("=", 1)
            elif ":" in layer_pair:
                student_layer, teacher_layer = layer_pair.split(":", 1)
            else:
                student_layer = teacher_layer = layer_pair
        elif isinstance(layer_pair, Sequence) and not isinstance(layer_pair, (bytes, bytearray)) and len(layer_pair) == 2:
            student_layer = str(layer_pair[0])
            teacher_layer = str(layer_pair[1])
        else:
            raise ValueError(
                "Each layerwise distillation pair must be a layer name or a 2-item sequence. "
                f"Got: {layer_pair!r}"
            )

        student_layer = student_layer.strip()
        teacher_layer = teacher_layer.strip()
        if not student_layer or not teacher_layer:
            raise ValueError(f"Invalid empty layer pair entry: {layer_pair!r}")
        parsed_pairs.append((student_layer, teacher_layer))
    return parsed_pairs


def resolve_layer_pairs(
    student_model: nn.Module,
    teacher_model: nn.Module,
    layer_pairs: Sequence[Any] | None = None,
    aligned_layers: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    parsed_pairs = parse_layer_pairs(layer_pairs or [])
    aligned_layers = aligned_layers or {}

    if aligned_layers:
        layer_list_name = aligned_layers["layer_list"].strip()
        output_module = aligned_layers.get("output_module")
        if output_module is not None:
            output_module = output_module.strip()

        teacher_lookup_model = teacher_model.ref_module if hasattr(teacher_model, "ref_module") else teacher_model
        student_layer_list = _resolve_submodule(student_model, layer_list_name)
        teacher_layer_list = _resolve_submodule(teacher_lookup_model, layer_list_name)

        student_child_names = [name for name, _module in student_layer_list.named_children()]
        teacher_child_names = [name for name, _module in teacher_layer_list.named_children()]
        if not student_child_names:
            raise ValueError(f"aligned_layers.layer_list '{layer_list_name}' has no child layers to align")
        if student_child_names != teacher_child_names:
            raise ValueError(
                f"Student/teacher child layers under '{layer_list_name}' do not match: "
                f"{student_child_names} vs {teacher_child_names}"
            )

        suffix = f".{output_module}" if output_module else ""
        parsed_pairs.extend(
            (f"{layer_list_name}.{child_name}{suffix}", f"{layer_list_name}.{child_name}{suffix}")
            for child_name in student_child_names
        )

    return parsed_pairs


@dataclass
class LayerwiseActivationCapture:
    student_model: nn.Module
    teacher_model: nn.Module
    layer_pairs: list[tuple[str, str]]

    def __post_init__(self):
        self._handles: list[Any] = []
        self._student_cache: dict[str, torch.Tensor] = {}
        self._teacher_cache: dict[str, torch.Tensor] = {}
        self._teacher_ref_cache: dict[str, torch.Tensor] = {}
        self._teacher_student_cache: dict[str, torch.Tensor] = {}
        self._phase: str | None = None
        self._use_trust_region_teacher = all(
            hasattr(self.teacher_model, attr) for attr in ("ref_module", "student_module", "mix_coef")
        )

        for student_layer_name, teacher_layer_name in self.layer_pairs:
            student_layer = _resolve_submodule(self.student_model, student_layer_name)
            self._handles.append(
                student_layer.register_forward_hook(self._make_student_hook(student_layer_name))
            )

            if self._use_trust_region_teacher:
                teacher_ref_layer = _resolve_submodule(self.teacher_model.ref_module, teacher_layer_name)
                teacher_student_layer = _resolve_submodule(self.teacher_model.student_module, teacher_layer_name)
                self._handles.append(
                    teacher_ref_layer.register_forward_hook(self._make_teacher_ref_hook(teacher_layer_name))
                )
                self._handles.append(
                    teacher_student_layer.register_forward_hook(self._make_teacher_student_hook(teacher_layer_name))
                )
            else:
                teacher_layer = _resolve_submodule(self.teacher_model, teacher_layer_name)
                self._handles.append(
                    teacher_layer.register_forward_hook(self._make_teacher_hook(teacher_layer_name))
                )

    def _make_student_hook(self, layer_name: str):
        def hook(_module, _inputs, output):
            if self._phase == "student":
                self._student_cache[layer_name] = _extract_activation_tensor(output)

        return hook

    def _make_teacher_hook(self, layer_name: str):
        def hook(_module, _inputs, output):
            if self._phase == "teacher":
                self._teacher_cache[layer_name] = _extract_activation_tensor(output)

        return hook

    def _make_teacher_ref_hook(self, layer_name: str):
        def hook(_module, _inputs, output):
            if self._phase == "teacher":
                self._teacher_ref_cache[layer_name] = _extract_activation_tensor(output)

        return hook

    def _make_teacher_student_hook(self, layer_name: str):
        def hook(_module, _inputs, output):
            if self._phase == "teacher":
                self._teacher_student_cache[layer_name] = _extract_activation_tensor(output)

        return hook

    def set_phase(self, phase: str | None) -> None:
        if phase not in {None, "student", "teacher"}:
            raise ValueError(f"Unsupported layerwise capture phase: {phase}")
        self._phase = phase

    def clear(self) -> None:
        self._student_cache.clear()
        self._teacher_cache.clear()
        self._teacher_ref_cache.clear()
        self._teacher_student_cache.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()

    def __enter__(self) -> "LayerwiseActivationCapture":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def get_student_activations(self) -> dict[str, torch.Tensor]:
        missing_layers = [student_layer for student_layer, _ in self.layer_pairs if student_layer not in self._student_cache]
        if missing_layers:
            raise ValueError(f"Missing student activations for layerwise distillation layers: {missing_layers}")
        return {student_layer: self._student_cache[student_layer] for student_layer, _ in self.layer_pairs}

    def get_teacher_activations(self) -> dict[str, torch.Tensor]:
        teacher_activations: dict[str, torch.Tensor] = {}
        for _student_layer, teacher_layer in self.layer_pairs:
            if self._use_trust_region_teacher:
                if teacher_layer not in self._teacher_ref_cache or teacher_layer not in self._teacher_student_cache:
                    raise ValueError(
                        f"Missing trust-region teacher activations for layer '{teacher_layer}' during layerwise distillation"
                    )
                teacher_activations[teacher_layer] = torch.lerp(
                    self._teacher_ref_cache[teacher_layer],
                    self._teacher_student_cache[teacher_layer],
                    float(self.teacher_model.mix_coef),
                )
            else:
                if teacher_layer not in self._teacher_cache:
                    raise ValueError(
                        f"Missing teacher activations for layer '{teacher_layer}' during layerwise distillation"
                    )
                teacher_activations[teacher_layer] = self._teacher_cache[teacher_layer]
        return teacher_activations
