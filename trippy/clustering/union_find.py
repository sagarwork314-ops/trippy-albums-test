"""Minimal disjoint-set (union-find) with path compression + union by rank."""
from __future__ import annotations


class UnionFind:
    def __init__(self, n: int):
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        parent = self._parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def groups(self) -> list[list[int]]:
        buckets: dict[int, list[int]] = {}
        for i in range(len(self._parent)):
            buckets.setdefault(self.find(i), []).append(i)
        return list(buckets.values())
