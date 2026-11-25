"""Incremental CPU inference with cache-policy-compatible request semantics."""

from collections import deque
import sys
from time import perf_counter_ns

import numpy as np
import torch

from features import CausalHistory, object_sequence
from lafc.policies.evict_value_v1 import EvictValueV1Policy


class Stream:
    def __init__(self, network, recompute=False):
        self.network = network
        self.recompute = recompute
        self.reset()

    def reset(self):
        self.state = None
        self.current = None
        self.history = deque(maxlen=self.network.config.request_history)
        self.inputs = deque(maxlen=self.network.config.request_history)
        self.pending = deque(maxlen=4)
        self.object_cache = {}
        self.step = 0

    @torch.inference_mode()
    def update(self, observation):
        self.pending.append(observation)
        order = self.network.config.order
        if order != "ordered":
            if self.step < 2:
                observation = np.zeros_like(observation)
            elif order == "permuted":
                observation = self.pending[-2 if self.step % 2 == 0 else -4]
            else:
                observation = self.pending[-3]
        self.step += 1
        tensor = torch.from_numpy(observation).view(1, 1, -1)
        if self.recompute:
            self.inputs.append(tensor)
            output, _ = self.network.global_outputs(torch.cat(tuple(self.inputs), dim=1))
            self.current = output[:, -1]
            self.history = deque(output[0].unbind(0), maxlen=self.network.config.request_history)
        else:
            output, self.state = self.network.global_outputs(tensor, self.state)
            self.current = output[:, -1]
            if self.network.attention or self.network.history_mlp:
                self.history.append(self.current[0])

    @torch.inference_mode()
    def score(self, features, residents, causal):
        x = torch.from_numpy(features).unsqueeze(0)
        mask = torch.ones(x.shape[:2], dtype=torch.bool)
        history = mask_history = objects = None
        if self.history:
            size = self.network.config.request_history
            history = self.current.new_zeros((1, size, self.current.shape[-1]))
            history[0, -len(self.history):] = torch.stack(tuple(self.history))
            mask_history = torch.arange(size).unsqueeze(0) >= size - len(self.history)
        if self.network.object_rnn and not self.recompute:
            missing = []
            cache = self.object_cache
            for i, item in enumerate(residents):
                last = causal.residents[item].last
                if item not in cache or cache[item][0] != last:
                    missing.append(i)
            if missing:
                updated = self.network.object_encode(*object_sequence(x[:, missing], self.network.config.object_history))
                for i, value in zip(missing, updated[0]):
                    item = residents[i]
                    cache[item] = causal.residents[item].last, value
            objects = torch.stack([cache[item][1] for item in residents]).unsqueeze(0)
        output = self.network.scores(x, mask, self.current, history, mask_history, objects)[0, :, 0].numpy()
        if not np.isfinite(output).all():
            raise FloatingPointError("Nonfinite scores in streaming replay")
        return output

    def discard(self, item):
        if item is not None:
            self.object_cache.pop(item, None)

    def memory(self):
        def size(t):
            return t.nelement() * t.element_size()
        recurrent = self.state if isinstance(self.state, tuple) else (self.state,)
        return sum(size(x) for x in recurrent if x is not None) + sum(size(x) for x in self.history) + sum(
            size(x) for _, x in self.object_cache.values())


class RequestPolicy(EvictValueV1Policy):
    """Use the published eviction/admission state transition and preserve oldest ties."""

    def __init__(self, model, recompute=False):
        super().__init__(scorer_mode="lightweight", history_window=64)
        self.stream = Stream(model, recompute)

    def reset(self, capacity, pages):
        super().reset(capacity, pages)
        self.causal = CausalHistory(capacity)
        self.stream.reset()
        self.measure = False
        self.feature_ns = 0
        self.update_ns = 0
        self.score_ns = 0
        self.latencies = []
        self.peak_state = 0
        self.peak_history_count = 0
        self.peak_metadata_estimate = 0

    def on_request(self, request):
        start = perf_counter_ns()
        current = self.causal.before(request.page_id, request.t, self.in_cache(request.page_id))
        features_end = perf_counter_ns()
        self.stream.update(current)
        update_end = perf_counter_ns()
        event = super().on_request(request)
        before_after = perf_counter_ns()
        self.causal.after(request.page_id, request.t, event.hit, event.evicted)
        self.stream.discard(event.evicted)
        if self.measure:
            self.feature_ns += features_end - start + perf_counter_ns() - before_after
            self.update_ns += update_end - features_end
            if event.evicted is not None:
                self.latencies.append(perf_counter_ns() - start)
            if request.t % 128 == 0:
                count = len(self.causal.ghosts) + len(self.causal.residents)
                self.peak_history_count = max(self.peak_history_count, count)
                self.peak_state = max(self.peak_state, self.stream.memory())
                entries = list(self.causal.residents.values()) + list(self.causal.ghosts.values())
                windows = list(self.causal.requests.values()) + list(self.causal.hits.values()) + [self.causal.evictions]
                shallow = (sys.getsizeof(self.causal.ghosts) + sys.getsizeof(self.causal.residents)
                           + sum(sys.getsizeof(e) + sys.getsizeof(e.gaps) for e in entries)
                           + sum(sys.getsizeof(w.items)+sys.getsizeof(w.count) for w in windows))
                self.peak_metadata_estimate = max(self.peak_metadata_estimate, shallow)
        return event

    def _choose_victim(self, request):
        residents = list(self._order)
        started = perf_counter_ns()
        features = self.causal.candidates(residents, request.t)
        before = perf_counter_ns()
        scores = self.stream.score(features, residents, self.causal)
        if self.measure:
            self.feature_ns += before - started
            self.score_ns += perf_counter_ns() - before
        return residents[int(np.argmax(scores))], {}
