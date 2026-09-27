import random
import numpy as np

from gptfuzzer.fuzzer import GPTFuzzer, PromptNode


class SelectPolicy:
    def __init__(self, fuzzer: GPTFuzzer):
        self.fuzzer = fuzzer

    def select(self) -> PromptNode:
        raise NotImplementedError("SelectPolicy must implement select().")

    def update(self, prompt_nodes: list[PromptNode]):
        pass


class RoundRobinSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: GPTFuzzer = None):
        super().__init__(fuzzer)
        self.index = 0

    def select(self) -> PromptNode:
        seed = self.fuzzer.prompt_nodes[self.index]
        seed.visited_num += 1
        return seed

    def update(self, prompt_nodes: list[PromptNode]):
        self.index = (self.index - 1 + len(self.fuzzer.prompt_nodes)) % len(self.fuzzer.prompt_nodes)


class RandomSelectPolicy(SelectPolicy):
    def select(self) -> PromptNode:
        seed = random.choice(self.fuzzer.prompt_nodes)
        seed.visited_num += 1
        return seed


class UCBSelectPolicy(SelectPolicy):
    def __init__(self, explore_coeff: float = 1.0, fuzzer: GPTFuzzer = None):
        super().__init__(fuzzer)
        self.step = 0
        self.last_choice_index = None
        self.explore_coeff = explore_coeff
        self.rewards = [0 for _ in range(len(self.fuzzer.prompt_nodes))]

    def select(self) -> PromptNode:
        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend([0] * (len(self.fuzzer.prompt_nodes) - len(self.rewards)))

        self.step += 1
        scores = np.zeros(len(self.fuzzer.prompt_nodes))

        for i, pn in enumerate(self.fuzzer.prompt_nodes):
            n = pn.visited_num + 1
            scores[i] = self.rewards[i] / n + self.explore_coeff * np.sqrt(
                2 * np.log(max(self.step, 2)) / n
            )

        self.last_choice_index = int(np.argmax(scores))
        seed = self.fuzzer.prompt_nodes[self.last_choice_index]
        seed.visited_num += 1
        return seed

    def update(self, prompt_nodes: list[PromptNode]):
        if self.last_choice_index is None:
            return
        succ_num = sum([p.num_jailbreak for p in prompt_nodes])
        self.rewards[self.last_choice_index] += succ_num / max(len(self.fuzzer.questions), 1)


class MCTSExploreSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: GPTFuzzer = None, ratio=0.5, alpha=0.1, beta=0.2):
        super().__init__(fuzzer)
        self.step = 0
        self.path = []
        self.last_choice_index = None
        self.rewards = []
        self.ratio = ratio
        self.alpha = alpha
        self.beta = beta

    def select(self) -> PromptNode:
        self.step += 1

        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend([0] * (len(self.fuzzer.prompt_nodes) - len(self.rewards)))

        self.path.clear()

        cur = max(
            self.fuzzer.initial_prompts_nodes,
            key=lambda pn: self.rewards[pn.index] / (pn.visited_num + 1)
            + self.ratio * np.sqrt(2 * np.log(max(self.step, 2)) / (pn.visited_num + 0.01)),
        )

        self.path.append(cur)

        while len(cur.child) > 0:
            if np.random.rand() < self.alpha:
                break

            cur = max(
                cur.child,
                key=lambda pn: self.rewards[pn.index] / (pn.visited_num + 1)
                + self.ratio * np.sqrt(2 * np.log(max(self.step, 2)) / (pn.visited_num + 0.01)),
            )
            self.path.append(cur)

        for pn in self.path:
            pn.visited_num += 1

        self.last_choice_index = cur.index
        return cur

    def update(self, prompt_nodes: list[PromptNode]):
        if self.last_choice_index is None or not prompt_nodes:
            return

        succ_num = sum([p.num_jailbreak for p in prompt_nodes])
        base = succ_num / max(len(self.fuzzer.questions) * len(prompt_nodes), 1)

        last_node = self.fuzzer.prompt_nodes[self.last_choice_index]

        for pn in reversed(self.path):
            self.rewards[pn.index] += base * max(self.beta, (1 - 0.1 * last_node.level))


class EXP3SelectPolicy(SelectPolicy):
    def __init__(self, gamma=0.05, alpha=25, fuzzer: GPTFuzzer = None):
        super().__init__(fuzzer)
        self.gamma = gamma
        self.alpha = alpha
        self.weights = [1.0 for _ in range(len(self.fuzzer.prompt_nodes))]
        self.probs = []
        self.last_choice_index = None

    def select(self) -> PromptNode:
        if len(self.fuzzer.prompt_nodes) > len(self.weights):
            self.weights.extend([1.0] * (len(self.fuzzer.prompt_nodes) - len(self.weights)))

        w = np.array(self.weights, dtype=float)
        probs = (1 - self.gamma) * w / w.sum() + self.gamma / len(self.fuzzer.prompt_nodes)

        self.last_choice_index = int(np.random.choice(len(self.fuzzer.prompt_nodes), p=probs))
        self.probs.append(probs[self.last_choice_index])

        seed = self.fuzzer.prompt_nodes[self.last_choice_index]
        seed.visited_num += 1
        return seed

    def update(self, prompt_nodes: list[PromptNode]):
        if self.last_choice_index is None or not prompt_nodes:
            return

        succ_num = sum([p.num_jailbreak for p in prompt_nodes])
        r = 1 - succ_num / len(prompt_nodes)

        prob = self.probs[-1] if self.probs else 1.0 / max(len(self.fuzzer.prompt_nodes), 1)
        x = -r / (prob + 1e-8)
        self.weights[self.last_choice_index] *= np.exp(self.alpha * x / max(len(self.fuzzer.prompt_nodes), 1))


class StrongJudgeGuidedSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: GPTFuzzer = None, exploration_ratio=0.25):
        super().__init__(fuzzer)
        self.exploration_ratio = exploration_ratio

    def select(self) -> PromptNode:
        if random.random() < self.exploration_ratio:
            seed = random.choice(self.fuzzer.prompt_nodes)
        else:
            seed = max(self.fuzzer.prompt_nodes, key=self._score)

        seed.visited_num += 1
        return seed

    def _score(self, pn: PromptNode) -> float:
        label = getattr(pn, "strong_judge_label", None)
        h = getattr(pn, "ra_heuristic", {}) or {}

        score = float(getattr(pn, "fitness_score", 0.0) or 0.0)

        if label == 2:
            score += 10.0
        elif label == 1:
            score += 5.0
        elif label == 0:
            score -= 4.0

        if h.get("clear_success_proxy_flag"):
            score += 3.0
        if h.get("partial_compliance_flag"):
            score += 1.5
        if h.get("risk_flag"):
            score += 1.0
        if h.get("refusal_flag"):
            score -= 2.0

        return score - 0.01 * getattr(pn, "visited_num", 0)


class OriginalASRPolicy:
    def __init__(self, fuzzer: GPTFuzzer):
        self.fuzzer = fuzzer

    def score(self, pn: PromptNode) -> float:
        h = getattr(pn, "ra_heuristic", {}) or {}

        score = float(getattr(pn, "original_score", 0.0) or 0.0) * 20.0

        if h.get("original_success_flag", False):
            score += 80.0
        if h.get("original_evaluator_success", False):
            score += 40.0
        if h.get("clear_success_proxy_flag"):
            score += 3.0
        if h.get("partial_compliance_flag"):
            score += 2.0
        if h.get("risk_flag"):
            score += 1.0

        if h.get("refusal_flag"):
            score -= 1.5
        if h.get("safe_redirect_flag"):
            score -= 1.0

        return score - 0.01 * getattr(pn, "visited_num", 0)


class StrongJudgePolicy:
    def __init__(self, fuzzer: GPTFuzzer):
        self.fuzzer = fuzzer

    def score(self, pn: PromptNode) -> float:
        label = getattr(pn, "strong_judge_label", None)
        h = getattr(pn, "ra_heuristic", {}) or {}

        score = float(getattr(pn, "raes_score", 0.0) or 0.0)
        score += float(getattr(pn, "fitness_score", 0.0) or 0.0)

        if label == 2:
            score += 18.0
        elif label == 1:
            score += 8.0
        elif label == 0:
            score -= 3.0

        if h.get("clear_success_proxy_flag"):
            score += 4.0
        if h.get("partial_compliance_flag"):
            score += 2.0
        if h.get("risk_flag"):
            score += 1.5
        if h.get("refusal_flag"):
            score -= 1.0
        if h.get("safe_redirect_flag"):
            score -= 0.5

        return score - 0.01 * getattr(pn, "visited_num", 0)


class DualPolicySelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: GPTFuzzer = None, alpha=0.5, retention_every=2, epsilon=0.0):
        super().__init__(fuzzer)
        self.original_policy = OriginalASRPolicy(fuzzer)
        self.judge_policy = StrongJudgePolicy(fuzzer)
        self.alpha = alpha
        self.retention_every = max(int(retention_every), 1)
        self.epsilon = max(0.0, min(float(epsilon), 1.0))
        self.step = 0

    def _original_success_nodes(self):
        nodes = []
        for pn in self.fuzzer.prompt_nodes:
            h = getattr(pn, "ra_heuristic", {}) or {}
            if h.get("original_success_flag", False) or h.get("original_evaluator_success", False):
                nodes.append(pn)
        return nodes

    def select(self) -> PromptNode:
        self.step += 1

        retained = self._original_success_nodes()
        if retained and self.step % self.retention_every == 0:
            seed = max(retained, key=lambda pn: self.original_policy.score(pn))
            seed.visited_num += 1
            return seed

        candidates = self.fuzzer.prompt_nodes
        scores = []

        for pn in candidates:
            original_s = self.original_policy.score(pn)
            judge_s = self.judge_policy.score(pn)
            score = self.alpha * original_s + (1.0 - self.alpha) * judge_s
            scores.append(score)

        # Epsilon exploration floor:
        # with probability epsilon, avoid deterministic argmax and explore a random parent.
        if self.epsilon > 0.0 and random.random() < self.epsilon:
            seed = random.choice(candidates)
        else:
            idx = int(np.argmax(scores))
            seed = candidates[idx]

        seed.visited_num += 1
        return seed


class RAESGuidedSelectPolicy(DualPolicySelectPolicy):
    """
    Compatibility wrapper.

    gptfuzz.py usually instantiates RAESGuidedSelectPolicy when
    --selection_policy raes is used. Here RAES is backed by the
    dual-policy selector so existing commands do not need changes.
    """

    def __init__(self, fuzzer: GPTFuzzer = None, alpha=None, retention_every=None, epsilon=None):
        import os

        if alpha is None:
            alpha = float(os.getenv("GPTFUZZ_DP_ALPHA", "0.5"))
        if retention_every is None:
            retention_every = int(os.getenv("GPTFUZZ_DP_RETENTION_EVERY", "2"))
        if epsilon is None:
            epsilon = float(os.getenv("GPTFUZZ_DP_EPSILON", "0.0"))

        super().__init__(
            fuzzer=fuzzer,
            alpha=alpha,
            retention_every=retention_every,
            epsilon=epsilon,
        )

class MCTSRAESSelectPolicy(MCTSExploreSelectPolicy):
    """
    MCTS backbone with RA/StrongJudge-aware reward shaping.
    Keeps GPTFuzzer's tree exploration and only augments path reward.
    """

    def __init__(
        self,
        fuzzer: GPTFuzzer = None,
        ratio=0.5,
        alpha=0.1,
        beta=0.2,
        ra_weight=0.35,
        label2_bonus=2.0,
        label1_bonus=0.8,
        partial_bonus=0.4,
        original_bonus=0.25,
        refusal_penalty=0.2,
        boundary_replay_ratio=0.25,
    ):
        super().__init__(fuzzer=fuzzer, ratio=ratio, alpha=alpha, beta=beta)
        self.ra_weight = ra_weight
        self.label2_bonus = label2_bonus
        self.label1_bonus = label1_bonus
        self.partial_bonus = partial_bonus
        self.original_bonus = original_bonus
        self.refusal_penalty = refusal_penalty
        self.boundary_replay_ratio = boundary_replay_ratio

    def _boundary_score(self, pn: PromptNode) -> float:
        label = getattr(pn, "strong_judge_label", None)
        h = getattr(pn, "ra_heuristic", {}) or {}

        score = 0.0
        if label == 2:
            score += 1.0
        elif label == 1:
            score += 5.0

        if h.get("partial_compliance_flag"):
            score += 3.0
        if h.get("clear_success_proxy_flag"):
            score += 2.0
        if h.get("risk_flag"):
            score += 1.0
        if h.get("original_success_flag", False) or h.get("original_evaluator_success", False):
            score += 0.5

        if h.get("refusal_flag"):
            score -= 0.8
        if h.get("safe_redirect_flag"):
            score -= 0.5

        return score - 0.02 * getattr(pn, "visited_num", 0)

    def _boundary_nodes(self):
        nodes = []
        for pn in self.fuzzer.prompt_nodes:
            label = getattr(pn, "strong_judge_label", None)
            h = getattr(pn, "ra_heuristic", {}) or {}
            if (
                label == 1
                or h.get("partial_compliance_flag")
                or h.get("clear_success_proxy_flag")
            ):
                if self._boundary_score(pn) > 0:
                    nodes.append(pn)
        return nodes

    def select(self) -> PromptNode:
        if random.random() < self.boundary_replay_ratio:
            nodes = self._boundary_nodes()
            if nodes:
                seed = max(nodes, key=self._boundary_score)
                seed.visited_num += 1
                self.last_choice_index = seed.index
                self.path = [seed]
                return seed

        return super().select()

    def _ra_bonus(self, prompt_nodes: list[PromptNode]) -> float:
        if not prompt_nodes:
            return 0.0

        total = 0.0
        for pn in prompt_nodes:
            label = getattr(pn, "strong_judge_label", None)
            h = getattr(pn, "ra_heuristic", {}) or {}

            if label == 2:
                total += self.label2_bonus
            elif label == 1:
                total += self.label1_bonus
            elif label == 0:
                total -= self.refusal_penalty

            if h.get("clear_success_proxy_flag"):
                total += self.partial_bonus
            if h.get("partial_compliance_flag"):
                total += self.partial_bonus * 0.5
            if h.get("original_success_flag", False) or h.get("original_evaluator_success", False):
                total += self.original_bonus
            if h.get("refusal_flag"):
                total -= self.refusal_penalty
            if h.get("safe_redirect_flag"):
                total -= self.refusal_penalty * 0.5

        return total / max(len(prompt_nodes), 1)

    def update(self, prompt_nodes: list[PromptNode]):
        if self.last_choice_index is None or not prompt_nodes:
            return

        succ_num = sum([p.num_jailbreak for p in prompt_nodes])
        base = succ_num / max(len(self.fuzzer.questions) * len(prompt_nodes), 1)
        shaped = base + self.ra_weight * self._ra_bonus(prompt_nodes)

        last_node = self.fuzzer.prompt_nodes[self.last_choice_index]

        for pn in reversed(self.path):
            self.rewards[pn.index] += shaped * max(self.beta, (1 - 0.1 * last_node.level))



class HybridRAESSelectPolicy(MCTSRAESSelectPolicy):
    """
    Hybrid RAES selector.

    Motivation:
    - MCTS-RAES is stable but weak.
    - DP-RAES has high upside but unstable jackpot/collapse behavior.
    - Hybrid-RAES combines stable MCTS exploration, DP top-k exploitation,
      and conservative elite-template replay.

    This class deliberately disables hard boundary replay from MCTSRAESSelectPolicy
    and uses portfolio-level mixing instead.
    """

    def __init__(self, fuzzer: GPTFuzzer = None):
        import os
        super().__init__(
            fuzzer=fuzzer,
            boundary_replay_ratio=0.0,
            ra_weight=float(os.getenv("GPTFUZZ_HYBRID_RA_WEIGHT", "0.35")),
            label2_bonus=float(os.getenv("GPTFUZZ_HYBRID_LABEL2_BONUS", "2.0")),
            label1_bonus=float(os.getenv("GPTFUZZ_HYBRID_LABEL1_BONUS", "0.8")),
            partial_bonus=float(os.getenv("GPTFUZZ_HYBRID_PARTIAL_BONUS", "0.4")),
            original_bonus=float(os.getenv("GPTFUZZ_HYBRID_ORIGINAL_BONUS", "0.25")),
            refusal_penalty=float(os.getenv("GPTFUZZ_HYBRID_REFUSAL_PENALTY", "0.2")),
        )

        self.hybrid_step = 0
        self.warmup_steps = int(os.getenv("GPTFUZZ_HYBRID_WARMUP_STEPS", "2"))

        self.mcts_ratio = float(os.getenv("GPTFUZZ_HYBRID_MCTS_RATIO", "0.50"))
        self.dp_ratio = float(os.getenv("GPTFUZZ_HYBRID_DP_RATIO", "0.30"))
        self.elite_ratio = float(os.getenv("GPTFUZZ_HYBRID_ELITE_RATIO", "0.20"))

        s = self.mcts_ratio + self.dp_ratio + self.elite_ratio
        if s <= 0:
            self.mcts_ratio, self.dp_ratio, self.elite_ratio = 0.5, 0.3, 0.2
        else:
            self.mcts_ratio /= s
            self.dp_ratio /= s
            self.elite_ratio /= s

        self.dp_alpha = float(os.getenv("GPTFUZZ_HYBRID_DP_ALPHA", "0.5"))
        self.dp_epsilon = float(os.getenv("GPTFUZZ_HYBRID_DP_EPSILON", "0.10"))
        self.dp_top_k = max(1, int(os.getenv("GPTFUZZ_HYBRID_DP_TOPK", "5")))
        self.elite_top_k = max(1, int(os.getenv("GPTFUZZ_HYBRID_ELITE_TOPK", "5")))

        self.hybrid_branch_counts = {
            "mcts": 0,
            "dp": 0,
            "elite": 0,
            "fallback": 0,
        }

    def _ensure_rewards(self):
        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend([0] * (len(self.fuzzer.prompt_nodes) - len(self.rewards)))

    def _path_to_root(self, seed: PromptNode):
        path = []
        cur = seed
        while cur is not None:
            path.append(cur)
            cur = getattr(cur, "parent", None)
        path.reverse()
        return path

    def _manual_select(self, seed: PromptNode, branch_name: str):
        self._ensure_rewards()
        self.step += 1
        self.path = self._path_to_root(seed)
        for pn in self.path:
            pn.visited_num += 1
        self.last_choice_index = seed.index
        self.hybrid_branch_counts[branch_name] = self.hybrid_branch_counts.get(branch_name, 0) + 1
        return seed

    def _original_score(self, pn: PromptNode) -> float:
        h = getattr(pn, "ra_heuristic", {}) or {}
        score = float(getattr(pn, "original_score", 0.0) or 0.0) * 20.0

        if h.get("original_success_flag", False):
            score += 80.0
        if h.get("original_evaluator_success", False):
            score += 40.0

        if h.get("clear_success_proxy_flag"):
            score += 3.0
        if h.get("partial_compliance_flag"):
            score += 2.0
        if h.get("risk_flag") or h.get("prompt_risk_flag"):
            score += 1.0

        if h.get("refusal_flag"):
            score -= 1.5
        if h.get("safe_redirect_flag"):
            score -= 1.0

        return score - 0.01 * getattr(pn, "visited_num", 0)

    def _semantic_score(self, pn: PromptNode) -> float:
        label = getattr(pn, "strong_judge_label", None)
        h = getattr(pn, "ra_heuristic", {}) or {}

        score = float(getattr(pn, "raes_score", 0.0) or 0.0)
        score += float(getattr(pn, "fitness_score", 0.0) or 0.0)

        if label == 2:
            score += 18.0
        elif label == 1:
            score += 8.0
        elif label == 0:
            score -= 3.0

        if h.get("clear_success_proxy_flag"):
            score += 4.0
        if h.get("partial_compliance_flag"):
            score += 2.0
        if h.get("risk_flag") or h.get("prompt_risk_flag"):
            score += 1.5

        if h.get("refusal_flag"):
            score -= 1.0
        if h.get("safe_redirect_flag"):
            score -= 0.5

        return score - 0.01 * getattr(pn, "visited_num", 0)

    def _dp_score(self, pn: PromptNode) -> float:
        return (
            self.dp_alpha * self._original_score(pn)
            + (1.0 - self.dp_alpha) * self._semantic_score(pn)
        )

    def _elite_score(self, pn: PromptNode) -> float:
        label = getattr(pn, "strong_judge_label", None)
        h = getattr(pn, "ra_heuristic", {}) or {}

        score = 0.0

        if label == 2:
            score += 10.0
        elif label == 1:
            score += 5.0

        if h.get("clear_success_proxy_flag"):
            score += 5.0
        if h.get("partial_compliance_flag"):
            score += 3.0
        if h.get("risk_flag") or h.get("prompt_risk_flag"):
            score += 0.5

        if h.get("original_success_flag", False) or h.get("original_evaluator_success", False):
            score += 0.5

        if h.get("refusal_flag"):
            score -= 3.0
        if h.get("safe_redirect_flag"):
            score -= 2.0

        return score - 0.05 * getattr(pn, "visited_num", 0)

    def _elite_nodes(self):
        nodes = []
        for pn in self.fuzzer.prompt_nodes:
            label = getattr(pn, "strong_judge_label", None)
            h = getattr(pn, "ra_heuristic", {}) or {}

            # Conservative elite filtering:
            # Do not replay label2 alone unless supported by heuristic evidence.
            is_supported_label2 = (
                label == 2
                and (
                    h.get("clear_success_proxy_flag")
                    or h.get("partial_compliance_flag")
                )
            )
            is_supported_label1 = (
                label == 1
                and (
                    h.get("clear_success_proxy_flag")
                    or h.get("partial_compliance_flag")
                )
            )
            is_proxy_elite = (
                h.get("clear_success_proxy_flag")
                and not h.get("refusal_flag")
                and not h.get("safe_redirect_flag")
            )

            if is_supported_label2 or is_supported_label1 or is_proxy_elite:
                if self._elite_score(pn) > 0:
                    nodes.append(pn)

        return nodes

    def _ranked_sample(self, nodes, score_fn, top_k: int):
        if not nodes:
            return None

        scores = np.array([score_fn(pn) for pn in nodes], dtype=float)
        k = min(max(int(top_k), 1), len(nodes))
        order = np.argsort(scores)[-k:][::-1]

        # rank-weighted sampling: 1, 1/2, 1/3, ...
        weights = np.array([1.0 / (rank + 1) for rank in range(len(order))], dtype=float)
        weights = weights / weights.sum()

        chosen_pos = int(np.random.choice(order, p=weights))
        return nodes[chosen_pos]

    def _select_dp(self):
        candidates = self.fuzzer.prompt_nodes
        if not candidates:
            return None

        if self.dp_epsilon > 0.0 and random.random() < self.dp_epsilon:
            return random.choice(candidates)

        return self._ranked_sample(candidates, self._dp_score, self.dp_top_k)

    def _select_elite(self):
        nodes = self._elite_nodes()
        if not nodes:
            return None
        return self._ranked_sample(nodes, self._elite_score, self.elite_top_k)

    def select(self) -> PromptNode:
        self.hybrid_step += 1

        # Warmup: preserve stable MCTS exploration before exploiting RAES signals.
        if self.hybrid_step <= self.warmup_steps:
            self.hybrid_branch_counts["mcts"] += 1
            return MCTSExploreSelectPolicy.select(self)

        r = random.random()

        # Branch 1: stable MCTS exploration.
        if r < self.mcts_ratio:
            self.hybrid_branch_counts["mcts"] += 1
            return MCTSExploreSelectPolicy.select(self)

        # Branch 2: DP-RAES top-k exploitation.
        if r < self.mcts_ratio + self.dp_ratio:
            seed = self._select_dp()
            if seed is not None:
                return self._manual_select(seed, "dp")

        # Branch 3: conservative elite replay.
        seed = self._select_elite()
        if seed is not None:
            return self._manual_select(seed, "elite")

        # Fallback: DP branch, then MCTS.
        seed = self._select_dp()
        if seed is not None:
            return self._manual_select(seed, "fallback")

        self.hybrid_branch_counts["mcts"] += 1
        return MCTSExploreSelectPolicy.select(self)
