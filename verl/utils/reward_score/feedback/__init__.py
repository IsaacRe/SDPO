import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from verl.utils.reward_score.feedback import math
from verl.utils.reward_score.feedback import code
from verl.utils.reward_score.feedback import gpqa
from verl.utils.reward_score.feedback import mcq
from verl.utils.reward_score.feedback import tooluse

MAX_BATCH_REWARD_WORKERS = 8


def _compute_single_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    max_test_cases: int | None = None,
) -> dict:
    if data_source in ["code", "livecodebench", "humanevalplus"]:
        results = code.compute_score(
            solution_str,
            ground_truth,
            extra_info,
            sparse_rewards=True,
            max_test_cases=max_test_cases,
        )
    elif data_source in ["math", "math500", "dapo_math", "gsm8k"]:
        results = math.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["gpqa"]:
        results = gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ["sciknoweval"]:
        results = mcq.compute_score(solution_str, ground_truth)
    elif data_source in ["tooluse"]:
        results = tooluse.compute_score(solution_str, ground_truth)
    else:
        raise ValueError(f"Reward style {data_source} not found.")
    return results


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: dict = None,
    max_test_cases: int | None = None,
    data_sources: list[str] | None = None,
    solution_strs: list[str] | None = None,
    ground_truths: list[str] | None = None,
    extra_infos: list[dict] | None = None,
) -> dict | list[dict]:
    if data_sources is not None:
        if solution_strs is None or ground_truths is None:
            raise ValueError("Batched reward scoring requires solution_strs and ground_truths.")

        if extra_infos is None:
            extra_infos = [None] * len(data_sources)

        batch_inputs = list(zip(data_sources, solution_strs, ground_truths, extra_infos, strict=True))
        if not batch_inputs:
            return []
        max_workers = min(len(batch_inputs), MAX_BATCH_REWARD_WORKERS)
        progress_enabled = os.environ.get("VERL_REWARD_PROGRESS", "0") == "1"
        progress_every = max(1, int(os.environ.get("VERL_REWARD_PROGRESS_EVERY", "1")))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _compute_single_score,
                    args[0],
                    args[1],
                    args[2],
                    extra_info=args[3],
                    max_test_cases=max_test_cases,
                ): idx
                for idx, args in enumerate(batch_inputs)
            }
            results = [None] * len(batch_inputs)
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                completed += 1
                if progress_enabled and (completed % progress_every == 0 or completed == len(batch_inputs)):
                    print(f"[reward] completed {completed}/{len(batch_inputs)} samples")
            return results

    if data_source is None or solution_str is None or ground_truth is None:
        raise ValueError("Reward scoring requires either batched inputs or single-example inputs.")

    return _compute_single_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info=extra_info,
        max_test_cases=max_test_cases,
    )
