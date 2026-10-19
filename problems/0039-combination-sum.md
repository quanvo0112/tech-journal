# 0039. Combination Sum

- **Problem Link:** https://leetcode.com/problems/combination-sum/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Backtracking`
- **LeetCode Topics:** `Array` / `Backtracking`
- **Core Pattern:** `Constrained Combination Enumeration / Backtracking with a Running Remainder`
- **Last Practiced:** 2026-10-19
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given an array of **distinct** integers `candidates` and a `target`, return all unique combinations that sum to `target`.
  - Each candidate may be reused **an unlimited number of times**.
  - Combinations are unordered: `[2, 2, 3]` and `[3, 2, 2]` count as the same answer.
  - Constraint: $1 \le \text{candidates.length} \le 30$, $2 \le \text{candidates[i]} \le 40$, $1 \le \text{target} \le 40$.
- **Core intuition:**
  - This is **constrained subset enumeration**: unlike Subsets, a branch is only accepted at a leaf when the running sum hits `target` exactly, so prune as early as possible.
  - **Maintain `remaining` instead of recomputing the sum:** pass `remaining = target - (sum so far)` down the recursion. It turns the acceptance test into an $O(1)$ comparison (`remaining == 0`) and makes the overflow/prune check trivial (`candidates[i] > remaining`).
  - **Reuse is expressed by recursing on the same index (`i`, not `i + 1`):** after picking `candidates[i]`, the next call may still pick `candidates[i]` again.
  - **The `start` index prevents order-duplicate combinations:** because the loop only offers indices $\ge start$ and never returns to earlier indices, a multiset like `{2, 2, 3}` is generated in exactly one non-decreasing order. Without `start` (or by always starting at `0`), `[2, 2, 3]`, `[2, 3, 2]`, `[3, 2, 2]` would all be produced.
  - **Sorting enables the prune:** once sorted, the first candidate exceeding `remaining` marks the end — every later candidate is at least as large, so `break` (not `continue`) is safe.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Naive Recursion without `start`, dedupe afterward ($O(\text{answer} \cdot \text{perms})$ time):**
   - At each step try every candidate with no index bound; whenever the sum equals `target`, store the combination.
   - *Drawback:* Explores every permutation of every combination, producing `[2,3,2]`, `[3,2,2]`, etc., then requires expensive set-based deduplication.

2. **Approach 2 — Dynamic Programming (Unbounded Knapsack) ($O(\text{target} \cdot N)$ DP table, plus output reconstruction):**
   - Build combinations bottom-up using `dp[t]` = list of combinations summing to `t`, reusing the standard unbounded-knapsack recurrence.
   - *Drawback:* Requires careful combination-list reconstruction and duplicate avoidance; heavier constant factors and more bookkeeping than the recursive enumeration.

3. **Approach 3 — Sorted Backtracking with `start` + remaining Sum (Chosen Optimal Solution):**
   - Sort `candidates` once.
   - `backtrack(start, remaining)`:
     - If `remaining == 0`: save `curr`, return.
     - For `i` from `start` to `N - 1`: if `candidates[i] > remaining`, `break`; else push `candidates[i]`, `backtrack(i, remaining - candidates[i])`, pop.
   - *Verdict:* Optimal. Every emitted subset is a distinct, correctly ordered combination, no post-processing dedupe, and the sort-based `break` abandons entire infeasible subtrees.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(N \log N + N^{\lceil T / M \rceil})$ where $M$ is the smallest candidate
  - Sorting the input costs $O(N \log N)$.
  - The recursion depth is bounded by $\lceil T / M \rceil$ (each pick reduces the remainder by at least $M$), and each level branches over at most $N$ candidates; the running-time bound is exponential in the worst case, proportional to the number of explored recursion nodes (which is bounded by the size of the answer enumeration).
- **Space Complexity:** $O(T / M)$ auxiliary / $O(\#\text{answers} \cdot T/M)$ total
  - `curr` and the recursion stack hold at most $\lceil T / M \rceil$ elements.
  - Output space stores all valid combinations, each of length at most $\lceil T / M \rceil$.

---

## 4. Edge Cases & Gotchas

- [x] **No valid combination:** `remaining` never reaches `0`; the function returns an empty result set.
- [x] **Candidate equals target (e.g. `[7]`, `target = 7`):** Picked once, `remaining` hits `0`, saved immediately.
- [x] **`break` vs `continue`:** Because the array is sorted, the first `candidates[i] > remaining` means all subsequent candidates also exceed `remaining`; using `continue` would still be correct but wastes iterations.
- [x] **Reuse index (`i` vs `i + 1`):** Recursing on `i` allows reuse; switching to `i + 1` silently converts this into the Combination Sum II rules.
- [x] **State leakage across calls:** `ans.clear()` and `curr.clear()` in the public method prevent stale data if the same `Solution` object is invoked twice.
- [x] **Unsorted input:** The `break` prune is only valid after `sort`; skipping the sort would terminate the loop early and drop valid combinations.
- **Common pitfall:** Recording `curr` when `remaining < 0` instead of relying on the `break`/`remaining == 0` guards produces invalid partial sums.

---

## 5. Clean Code (Optimal Solution: Backtracking)

```cpp
class Solution {
private:
    vector<vector<int>> ans;
    vector<int> curr;

    void backtrack(int start, int remaining,
                   vector<int>& candidates) {
        if (remaining == 0) {
            ans.push_back(curr);
            return;
        }

        for (int i = start; i < (int)candidates.size(); ++i) {
            // Sorted array: all later candidates are >= this one, so stop.
            if (candidates[i] > remaining) {
                break;
            }

            curr.push_back(candidates[i]);

            // Recurse on i (not i + 1) to allow reusing the same candidate.
            backtrack(i, remaining - candidates[i], candidates);

            curr.pop_back();
        }
    }

public:
    vector<vector<int>> combinationSum(vector<int>& candidates,
                                       int target) {
        sort(candidates.begin(), candidates.end());

        ans.clear();
        curr.clear();

        backtrack(0, target, candidates);

        return ans;
    }
};
```

---

## 6. Review & Takeaways

### Visual Recursion Tree ($candidates = [2, 3, 6, 7]$, $target = 7$)

```text
backtrack(start=0, remaining=7)
├── pick 2: backtrack(0, 5)
│   ├── pick 2: backtrack(0, 3)
│   │   ├── pick 2: backtrack(0, 1)  -> 2 > 1, break
│   │   └── pick 3: backtrack(1, 0)  save [2,2,3]
│   └── pick 3: backtrack(1, 2)      -> 3 > 2, break
├── pick 3: backtrack(1, 4)
│   └── pick 3: backtrack(1, 1)      -> 3 > 1, break
├── pick 6: backtrack(2, 1)          -> 6 > 1, break
└── pick 7: backtrack(3, 0)  save [7]

Final Result: [[2, 2, 3], [7]]
```

---

### The Architectural Mental Model

```text
                     Combination Sum (target)
                               ↓
                     Sort + Backtracking (start)
                               ↓
                 Track running remaining = target - sum
                               ↓
              ┌────────────────┴────────────────┐
              ↓                                 ↓
       remaining == 0                    candidates[i] > remaining
              ↓                                 ↓
        save curr, return                      break loop
              ↓
        push candidates[i]
              ↓
        recurse (i, remaining - candidates[i])   ── reuse index i
              ↓
        pop_back() ──> try next sibling
```

* **Next Review Date:**
* **Key Takeaway:** Carry a `remaining` counter, recurse on the **same** index `i` to allow reuse, and `break` on the first candidate exceeding `remaining` — the sorted array then guarantees each combination appears exactly once.
