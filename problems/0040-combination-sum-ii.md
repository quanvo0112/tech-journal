# 0040. Combination Sum II

- **Problem Link:** https://leetcode.com/problems/combination-sum-ii/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Backtracking`
- **LeetCode Topics:** `Array` / `Backtracking`
- **Core Pattern:** `Sorted Backtracking with Same-Level Duplicate Skipping`
- **Last Practiced:** 2026-10-20
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given `candidates` (which **may contain duplicates**) and a `target`, return all **unique** combinations summing to `target`.
  - Each element may be used **at most once** per combination.
  - Combinations are unordered: `[1, 7]` and `[7, 1]` are the same answer.
  - Constraint: $1 \le \text{candidates.length} \le 100$, $1 \le \text{candidates[i]} \le 50$, $1 \le \text{target} \le 30$.
- **Core intuition:**
  - Two independent constraints must be enforced: **no reuse of an index** and **no duplicate combinations**. Fixing one does not fix the other.
  - **No reuse $\Rightarrow$ recurse on `i + 1`.** Contrast with Combination Sum, which recurses on `i`. This single character is the entire semantic difference between the two problems.
  - **Duplicates in the input are the real trap.** Since the same value can appear at several indices, two distinct index paths can yield the same multiset (e.g. picking the first `1` or the second `1`). The index-level `i + 1` rule does *not* prevent this.
  - **Sort first.** Sorting clusters equal values together so that duplicate branches are adjacent and can be detected with a single comparison.
  - **Skip duplicates at the same recursion level, not globally:** `i > start && candidates[i] == candidates[i - 1]`. The `i > start` guard is critical — it means "this value was already tried as a *sibling* at this depth, so skip it", while still allowing the same value to be picked again deeper (from a different index) when it is genuinely a second copy.
  - **The invariant:** at any recursion depth, the loop picks at most one representative per distinct value. Therefore each multiset is generated exactly once, in non-decreasing index order.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Enumerate all subsets, filter by sum, dedupe with a set ($O(2^N \cdot N)$ plus hashing):**
   - Generate every subset, keep the ones summing to `target`, insert into `set<vector<int>>` to remove duplicates.
   - *Drawback:* Explores all $2^N$ subsets even though almost all are infeasible, and pays the cost of constructing/hashing full vectors. Does not scale and hides the pattern.

2. **Approach 2 — Backtracking without sorting or duplicate skipping:**
   - Recurse on `i + 1` to respect single use, but ignore repeated values.
   - *Drawback:* With `[1, 1, 2, 5, 6, 7, 10]`, the two `1`s are distinct indices, so `[1, 7]` gets emitted twice (once per `1`). Post-hoc deduplication becomes mandatory.

3. **Approach 3 — Sorted Backtracking + Same-Level Duplicate Skip (Chosen Optimal Solution):**
   - Sort `candidates` ascending.
   - `backtrack(start, remaining)`:
     - If `remaining == 0`: save `curr`, return.
     - For `i` from `start` to `N - 1`:
       - If `i > start && candidates[i] == candidates[i - 1]`: `continue` (duplicate sibling — skip).
       - If `candidates[i] > remaining`: `break` (sorted, so all later values are too large).
       - Push, `backtrack(i + 1, remaining - candidates[i])`, pop.
   - *Verdict:* Optimal. Duplicates are pruned at the source instead of filtered afterward; the sorted order additionally enables the `break` pruning. Every emitted combination is unique by construction — no set, no sorting of results.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(N \log N + N \cdot 2^N)$ in the worst case
  - Sorting costs $O(N \log N)$.
  - The recursion explores at most $2^N$ index subsets, and each node scans up to $N$ candidates in its loop (duplicate skips reduce the constant but not the bound). In practice pruning by `remaining` cuts this dramatically.
  - Copying `curr` into `ans` costs $O(k)$ per emitted combination, absorbed into the output-sensitive term.
- **Space Complexity:** $O(N)$ auxiliary, plus output
  - `curr` and the recursion stack each hold at most $N$ elements (depth is bounded by the number of picks, not by `target`, since each value is consumed once).
  - `ans` stores all valid combinations; total output size is $O(\#\text{answers} \cdot k)$ where $k \le N$.

---

## 4. Edge Cases & Gotchas

- [x] **All elements identical** (`[2, 2, 2]`, `target = 4`): The `i > start` guard lets `[2, 2]` form at depth 2 (two separate indices) while ensuring it is emitted only once.
- [x] **Duplicate values far apart after sorting?** Impossible — sorting places them adjacent, which is why comparing against `candidates[i - 1]` is sufficient.
- [x] **`i > start` vs `i > 0`:** Using `i > 0` would be wrong; it would also skip a value that legitimately appears as the first pick at a deeper level. The comparison must be relative to the current loop's `start`.
- [x] **`break` vs `continue` for the prune:** With sorted input the first `candidates[i] > remaining` implies all later values exceed `remaining`; `break` is both correct and faster.
- [x] **Recursing with `i + 1`, not `i`:** Using `i` silently re-enables unbounded reuse and turns the problem into Combination Sum (producing wrong answers such as `[2, 2]` from a single `2`).
- [x] **No valid combination:** Returns an empty vector.
- [x] **State leakage across calls:** `ans.clear()` / `curr.clear()` in the public method keep the object reusable.
- [x] **Overflow:** `remaining` never goes negative because of the `break` guard; `int` is more than sufficient for the given constraints.
- **Common pitfall:** Skipping duplicates without first sorting — adjacent comparison is meaningless on unsorted input.

---

## 5. Clean Code (Optimal Solution: Sorted Backtracking)

```cpp
class Solution {
private:
    vector<vector<int>> ans;
    vector<int> curr;

    void backtrack(int start, int remaining, vector<int>& candidates) {
        if (remaining == 0) {
            ans.push_back(curr);
            return;
        }

        for (int i = start; i < (int)candidates.size(); ++i) {
            // Skip duplicate values at the same recursion level.
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }

            // Sorted array: all following values are also too large.
            if (candidates[i] > remaining) {
                break;
            }

            curr.push_back(candidates[i]);

            // i + 1 ensures each element is used at most once.
            backtrack(i + 1, remaining - candidates[i], candidates);

            curr.pop_back();
        }
    }

public:
    vector<vector<int>> combinationSum2(vector<int>& candidates,
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

### Visual Recursion Tree ($candidates = [1, 1, 2, 5, 6, 7, 10]$, $target = 8$)

```text
backtrack(start=0, remaining=8)
├── pick 1 (idx 0): backtrack(1, 7)
│   ├── idx 1 = 1  -> duplicate of idx 0 at same level: continue
│   ├── pick 2 (idx 2): backtrack(3, 5)
│   │   ├── pick 5 (idx 3): backtrack(4, 0)  save [1, 2, 5]
│   │   ├── 6 > 5 -> break
│   ├── 5 > 7? no: pick 5 (idx 3): backtrack(4, 2)
│   │   └── 6 > 2 -> break
│   ├── pick 6 (idx 4): backtrack(5, 1)
│   │   └── 7 > 1 -> break
│   └── pick 7 (idx 5): backtrack(6, 0)      save [1, 7]
├── pick 1 (idx 1) -> duplicate at same level: continue
├── pick 2 (idx 2): backtrack(3, 6)
│   ├── pick 5 (idx 3): backtrack(4, 1)
│   │   └── 6 > 1 -> break
│   └── pick 6 (idx 4): backtrack(5, 0)      save [2, 6]
├── pick 5 (idx 3): backtrack(4, 3)
│   └── 6 > 3 -> break
├── pick 6 (idx 4): backtrack(5, 2)
│   └── 7 > 2 -> break
├── pick 7 (idx 5): backtrack(6, 1)
│   └── 10 > 1 -> break
└── 10 > 8 -> break

Final Result: [[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]]
```

Note that `[1, 1, 6]` is produced through **two different indices** (`idx 0` then `idx 1`), while `[1, 7]` is produced exactly once — the second `1` is skipped only when it is a *sibling* of the first.

---

### The Architectural Mental Model

```text
                    Combination Sum II (unique, no reuse)
                                   ↓
                          Sort candidates ascending
                                   ↓
                   Backtrack(start, remaining) over indices
                                   ↓
        ┌──────────────────────────┼──────────────────────────┐
        ↓                          ↓                          ↓
 i > start &&                candidates[i]             otherwise:
 candidates[i] ==               > remaining                pick
 candidates[i-1]                     ↓                       ↓
        ↓                        break loop          recurse(i + 1, ...)
     continue                  (sorted prune)               ↓
                                                         pop_back()
```

### Difference from Combination Sum (0039)

| Aspect | Combination Sum (0039) | Combination Sum II (0040) |
|---|---|---|
| Reuse an element | Yes | No |
| Recursive call | `backtrack(i, ...)` | `backtrack(i + 1, ...)` |
| Duplicate handling | Not needed (distinct input) | Skip `candidates[i] == candidates[i - 1]` when `i > start` |
| Sort required | Only for the `break` prune | Required for both dedupe and prune |
| Unique answers | Guaranteed by distinct input | Guaranteed by same-level skip |

* **Next Review Date:**
* **Key Takeaway:** Sort first, recurse on `i + 1` to consume each element once, and skip an equal sibling with `i > start` — that single guard distinguishes a legitimate second copy of a value from a duplicate branch.
