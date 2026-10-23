# 0090. Subsets II

- **Problem Link:** https://leetcode.com/problems/subsets-ii/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Backtracking`
- **LeetCode Topics:** `Array` / `Backtracking` / `Bit Manipulation`
- **Core Pattern:** `Sorted Backtracking with Same-Level Duplicate Skipping`
- **Last Practiced:** 2026-10-23
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given an integer array `nums` that **may contain duplicates**, return all possible subsets (the power set) **without duplicate subsets**.
  - The solution set must not contain duplicated subsets; the output order is arbitrary.
  - Constraint: $1 \le \text{nums.length} \le 10$, $-10 \le \text{nums[i]} \le 10$.
- **Core intuition:**
  - **This is Subsets (0078) plus one new constraint.** The enumeration skeleton — record `path` at the top of every call, loop from `start`, recurse on `i + 1` — is carried over unchanged.
  - **Why the naive extension fails:** duplicated values live at distinct indices. With `nums = [1, 2, 2]`, the index paths `(1, 2a)` and `(1, 2b)` produce the same multiset `[1, 2]`, and `(nothing)` at the root vs. nothing else is fine, but `[2]` likewise appears twice. Index-level `i + 1` dedupes *positions*, not *values*.
  - **Sort first:** equal values become adjacent, so a duplicate branch can be detected with a single comparison to `nums[i - 1]`.
  - **Skip duplicates at the same recursion level, not globally:** `i > start && nums[i] == nums[i - 1]`.
    - `i > start` — this value was already offered as a **sibling** at the current depth, so this branch is redundant.
    - Dropping the guard (using `i > 0`) would wrongly skip a legitimate second copy deeper in the tree, and `[2, 2]` would disappear.
  - **The invariant:** at any recursion level the loop picks at most one representative per distinct value. Therefore each subset is generated exactly once, and every occurrence of a value is consumed in left-to-right index order.
  - **Contrast with Combination Sum II (0040):** identical duplicate-skipping rule, different termination. Here *every* node is a valid answer (no `remaining` budget), so we record before the loop and never prune on sum.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Generate all subsets, then dedupe with a set ($O(2^n \cdot n)$ plus hashing):**
   - Run the plain Subsets backtracking, then insert every subset into `set<vector<int>>` (or sort + `unique`) before returning.
   - *Drawback:* Still explores all $2^n$ branches including the duplicate ones, then pays the cost of constructing and hashing full vectors. Memory and time are wasted on output that is thrown away.

2. **Approach 2 — Count frequencies and recurse on $(value, count)$ pairs:**
   - Build a frequency map, then for each distinct value choose to take `0..count` copies.
   - *Drawback:* Correct and asymptotically similar, but requires an extra map pass and index bookkeeping; the loop-with-skip form is shorter and generalizes directly from Subsets.

3. **Approach 3 — Sorting + Backtracking with Same-Level Duplicate Skip (Chosen Optimal Solution):**
   - `sort(nums.begin(), nums.end())`.
   - `backtrack(start)`:
     - `res.push_back(path)` — every entry is a valid subset (this emits `[]` at the root).
     - For `i` from `start` to `n - 1`:
       - If `i > start && nums[i] == nums[i - 1]`: `continue` (duplicate sibling).
       - `path.push_back(nums[i])`, `backtrack(i + 1)`, `path.pop_back()`.
   - *Verdict:* Optimal and idiomatic. Duplicates are pruned **at the source** rather than filtered afterwards; no set, no result sorting, and the only delta from Subsets is the sort plus the two-condition guard.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(n \log n + n \cdot 2^n)$ in the worst case
  - Sorting costs $O(n \log n)$.
  - In the best case (all distinct) the tree has $2^n$ nodes; deduplication only removes branches, so $2^n$ remains an upper bound on emitted subsets.
  - Each emitted subset is copied into `res` in up to $O(n)$ time, giving the $n \cdot 2^n$ output-sensitive term. The loop scan per node is absorbed into that bound.
- **Space Complexity:** $O(n)$ auxiliary, plus output
  - `path` and the recursion stack each hold at most $n$ elements (depth is bounded by the number of picks).
  - `res` stores all unique subsets; total output size is $O(\#\text{answers} \cdot k)$ with $k \le n$, which is not counted as auxiliary space.

---

## 4. Edge Cases & Gotchas

- [x] **Empty subset:** The root call records `path` while empty, so `[]` is always emitted exactly once.
- [x] **All elements identical** (`[2, 2, 2]`): Produces `[]`, `[2]`, `[2, 2]`, `[2, 2, 2]`. The guard allows the copies to stack deeper while suppressing sibling repeats.
- [x] **Single element (`n = 1`):** Returns `[[], [nums[0]]]`.
- [x] **Negative values:** Sorting and equality comparisons are value-based; negatives behave identically. No arithmetic is performed, so no overflow concerns (constraints are tiny anyway).
- [x] **`i > start` vs `i > 0`:** The comparison must be relative to the current loop's `start`. Using `i > 0` would skip a value that legitimately begins a deeper branch, silently dropping subsets such as `[2, 2]`.
- [x] **Forgetting to sort:** Adjacent comparison is meaningless on unsorted input, so duplicate branches survive and the output is wrong.
- [x] **Filtering duplicates globally:** Skipping every `nums[i] == nums[i - 1]` unconditionally would erase `[2, 2]`; only *same-level* duplicates are redundant.
- [x] **State leakage across calls:** `res.clear()` and `path.clear()` in the public method keep the `Solution` object reusable.
- **Common pitfall:** Trying to insert into a set for deduplication instead of understanding *why* sibling branches collide — the guard is both faster and the reusable mental model.

---

## 5. Clean Code (Optimal Solution: Sorted Backtracking)

```cpp
class Solution {
private:
    vector<vector<int>> res;
    vector<int> path;

    void backtrack(vector<int>& nums, int start) {
        // Every entry into backtrack() is already a valid subset.
        res.push_back(path);

        for (int i = start; i < (int)nums.size(); ++i) {
            // Skip duplicate values at the same recursion level.
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }

            path.push_back(nums[i]);

            // i + 1 ensures each index is used at most once per subset.
            backtrack(nums, i + 1);

            path.pop_back();  // Backtrack
        }
    }

public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());

        res.clear();
        path.clear();

        backtrack(nums, 0);

        return res;
    }
};
```

---

## 6. Review & Takeaways

### Visual Recursion Tree ($nums = [1, 2, 2]$)

```text
backtrack(0) save []
├── pick 1 (idx 0): backtrack(1) save [1]
│   ├── pick 2 (idx 1): backtrack(2) save [1,2]
│   │   └── pick 2 (idx 2): backtrack(3) save [1,2,2]
│   └── idx 2 = 2 -> duplicate sibling (i > start): continue
├── pick 2 (idx 1): backtrack(2) save [2]
│   └── pick 2 (idx 2): backtrack(3) save [2,2]
└── idx 2 = 2 -> duplicate sibling (i > start): continue

Final Result: [], [1], [1,2], [1,2,2], [2], [2,2]
```

The second `2` is skipped only where it would duplicate a *sibling choice*; deeper in the `[1, ...]` and `[2, ...]` branches it is still picked, which is exactly what makes `[1,2,2]` and `[2,2]` reachable.

---

### The Architectural Mental Model

```text
                        Subsets II (duplicates allowed)
                                    ↓
                          Sort nums ascending
                                    ↓
                    Backtrack(start) over indices
                                    ↓
                 Record path at every node (valid subset)
                                    ↓
              ┌─────────────────────┴─────────────────────┐
              ↓                                           ↓
  i > start && nums[i] == nums[i - 1]              otherwise: pick nums[i]
              ↓                                           ↓
          continue                              recurse(i + 1)
       (duplicate sibling)                            ↓
                                                  pop_back()
```

### Difference from Subsets (0078)

| Aspect | Subsets (0078) | Subsets II (0090) |
|---|---|---|
| Input | Distinct elements | May contain duplicates |
| Sorting | Not required | Required (clusters duplicates) |
| Loop guard | None | `i > start && nums[i] == nums[i - 1]` |
| Unique output | Guaranteed by distinct input | Guaranteed by same-level skip |
| Record point | Top of every call | Top of every call (unchanged) |

* **Next Review Date:**
* **Key Takeaway:** Subsets II is Subsets plus one line of defense — sort, then skip an equal sibling with `i > start && nums[i] == nums[i - 1]`, which prunes duplicate branches at the source while still allowing genuine repeated values to stack.
