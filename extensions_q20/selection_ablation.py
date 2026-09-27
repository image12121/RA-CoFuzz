"""Strictly disable one HybridRAES selection branch without editing legacy source."""
from contextlib import contextmanager


BRANCHES = ("mcts", "dp", "elite")


@contextmanager
def strict_branch_ablation(disabled_branch, selection_module=None):
    if disabled_branch not in BRANCHES:
        raise ValueError("invalid_selection_ablation_config")

    if selection_module is None:
        from gptfuzzer.fuzzer import selection as selection_module

    cls = selection_module.HybridRAESSelectPolicy
    original_select = cls.select
    observation = {
        "disabled_branch": disabled_branch,
        "selection_attempts": {name: 0 for name in BRANCHES},
        "selection_counts": {name: 0 for name in BRANCHES},
        "fallback_events": 0,
    }
    last_selector = [None]

    def choose(self, branch):
        observation["selection_attempts"][branch] += 1
        if branch == "mcts":
            self.hybrid_branch_counts["mcts"] += 1
            observation["selection_counts"]["mcts"] += 1
            return selection_module.MCTSExploreSelectPolicy.select(self)
        seed = self._select_dp() if branch == "dp" else self._select_elite()
        if seed is None:
            return None
        observation["selection_counts"][branch] += 1
        return self._manual_select(seed, branch)

    def strict_select(self):
        last_selector[0] = self
        self.hybrid_step += 1
        allowed = [name for name in BRANCHES if name != disabled_branch]

        if self.hybrid_step <= self.warmup_steps:
            if "mcts" not in allowed:
                raise ValueError("invalid_selection_ablation_config")
            return choose(self, "mcts")

        weights = {
            "mcts": self.mcts_ratio,
            "dp": self.dp_ratio,
            "elite": self.elite_ratio,
        }
        active = [(name, weights[name]) for name in allowed if weights[name] > 0.0]
        total = sum(weight for _, weight in active)
        if total <= 0.0:
            raise ValueError("invalid_selection_ablation_config")
        draw = selection_module.random.random() * total
        cumulative = 0.0
        selected = active[-1][0]
        for name, weight in active:
            cumulative += weight
            if draw < cumulative:
                selected = name
                break

        result = choose(self, selected)
        if result is not None:
            return result

        observation["fallback_events"] += 1
        # Preserve the legacy fallback preference while excluding the ablated branch.
        for name in ("elite", "dp", "mcts"):
            if name == selected or name not in allowed:
                continue
            result = choose(self, name)
            if result is not None:
                return result
        raise RuntimeError("selection_ablation_no_candidate")

    cls.select = strict_select
    try:
        yield observation
    finally:
        cls.select = original_select
        selector = last_selector[0]
        if selector is not None:
            observation["native_branch_counts"] = dict(
                getattr(selector, "hybrid_branch_counts", {}))
            observation["hybrid_steps"] = int(getattr(selector, "hybrid_step", 0))
        observation["disabled_branch_attempts"] = observation[
            "selection_attempts"][disabled_branch]
        observation["disabled_branch_selections"] = observation[
            "selection_counts"][disabled_branch]
