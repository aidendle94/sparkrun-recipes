"""Per-request prompt-lookup index for the hybrid DSpark drafter.

Maps every n-token window of a request's sequence (prompt + committed output) to the position that followed its most
recent occurrence. `propose(k)` returns the up-to-k tokens that followed the last earlier occurrence of the sequence's
final n tokens, or an empty list when that n-gram has not occurred before. Updating costs O(1) per committed token and a
proposal O(k), so the index can be maintained on the scheduler thread every decode step.

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations


class LookupIndex:
    __slots__ = ("n", "tokens", "_follow")

    def __init__(self, n: int = 4, tokens=()):
        self.n = n
        self.tokens: list[int] = []
        self._follow: dict[tuple, int] = {}
        self.extend(tokens)

    def extend(self, new_tokens) -> None:
        """Append committed tokens. The window ending just before each new token gains that token as its follower."""
        toks, n, follow = self.tokens, self.n, self._follow
        for t in new_tokens:
            j = len(toks)                        # index the appended token will occupy
            if j >= n:
                follow[tuple(toks[j - n:j])] = j
            toks.append(int(t))

    def propose(self, k: int) -> list[int]:
        """Up to k tokens that followed the most recent EARLIER occurrence of the last n tokens (none if unseen)."""
        toks, n = self.tokens, self.n
        if len(toks) < n or k <= 0:
            return []
        f = self._follow.get(tuple(toks[len(toks) - n:]))
        if f is None:
            return []
        return toks[f:f + k]
