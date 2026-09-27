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
    def __init__(self, fuzzer: GPTFuzzer = None, alpha=0.5, retention_every=2):
        super().__init__(fuzzer)
        self.original_policy = OriginalASRPolicy(fuzzer)
        self.judge_policy = StrongJudgePolicy(fuzzer)
        self.alpha = alpha
        self.retention_every = max(int(retention_every), 1)
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

    def __init__(self, fuzzer: GPTFuzzer = None, alpha=0.5, retention_every=2):
        super().__init__(fuzzer=fuzzer, alpha=alpha, retention_every=retention_every)
