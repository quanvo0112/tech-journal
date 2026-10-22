# 0046. Permutations

- **Problem Link:** https://leetcode.com/problems/permutations/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Backtracking`
- **LeetCode Topics:** `Array` / `Backtracking`
- **Core Pattern:** `Permutation Enumeration / Backtracking with a used[] Set`
- **Last Practiced:** 2026-10-22
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given an array `nums` of **distinct** integers, return **all possible permutations** in any order.
  - Constraint: $1 \le \text{nums.length} \le 6$, all elements unique.
  - Return type is `vector<vector<int>>`; order of the output does not matter.
- **Core intuition:**
  - Unlike Subsets, **order matters here**: `[1, 2, 3]` and `[3, 2, 1]` are different answers. So the `start` index trick from Subsets/Combination Sum is wrong — it would forbid revisiting earlier positions and emit only the sorted order.
  - With `n` distinct values there are exactly $n!$ permutations, and every permutation must be built by choosing **any unused element** at each depth.
  - **`used[i]` is the membership invariant:** at any recursion depth, `path` is a sequence of already-chosen indices and `used` marks them. The loop offers every index whose `used[i] == false`, so each value appears exactly once per permutation and no index is repeated.
  - **Acceptance test is positional, not value-based:** a branch is complete the moment `path.size() == nums.size()`. No sum/target/constraint check is needed — every leaf of the tree is a valid answer.
  - **`path` is shared and mutated in place:** `push_back` -> recurse -> `pop_back` reuses a single buffer, while `used` is toggled symmetrically. Forgetting `used[i] = false` after `pop_back` silently prunes whole subtrees of valid permutations.
  - **Why not a `set` for membership?** A `vector<bool>` indexed by position is $O(1)$ with no hashing, no allocation per depth, and directly matches "this index was used". Since elements are distinct, index-level tracking is equivalent to value-level tracking.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Generate all $n$-tuples, then filter ($O(n^n \cdot n!)$ time):**
   - Recursively build every length-$n$ sequence from all `n` values, then discard sequences containing duplicate indices.
   - *Drawback:* Explores $n^n$ candidates to keep only $n!$ of them; almost all work is thrown away, and it breaks down entirely once duplicates appear in the input.

2. **Approach 2 — In-place Swap Backtracking ($O(n \cdot n!)$ time, $O(n)$ auxiliary):**
   - At depth `i`, swap `nums[i]` with each `nums[j]` for $j \ge i$, recurse on `i + 1`, then swap back. When `i == n`, record `nums`.
   - *Verdict:* Slightly leaner (no `used` array) and the standard in-place trick, but it **mutates the caller's input**, relies on symmetric swaps that are easy to get wrong, and does not generalize to Permutations II (duplicates) without an extra sort + skip rule.

3. **Approach 3 — Backtracking with a `used` Array (Chosen Optimal Solution):**
   - `backtrack()`:
     - If `path.size() == nums.size()`: save `path`, return.
     - For each `i`: if `used[i]`, skip; else set `used[i] = true`, `path.push_back(nums[i])`, recurse, `path.pop_back()`, `used[i] = false`.
   - *Verdict:* Optimal and the most transferable template. The explicit `used` state makes the "already placed at this depth" condition obvious, never mutates the input, and extends directly to Permutations II by adding `sort` + the duplicate-skip rule (`i > 0 && nums[i] == nums[i-1] && !used[i-1]`). It also carries over to Combination Sum style problems once the loop bound is swapped for `start`.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(n \cdot n!)$
  - There are $n!$ complete permutations, and each one is copied into `res` in $O(n)$ time, giving the $n \cdot n!$ term.
  - The recursion tree has $\sum_{k=0}^{n} \frac{n!}{(n-k)!} \approx e \cdot n!$ nodes with $O(1)$ work per node, which is dominated by the copying cost.
- **Space Complexity:** $O(n \cdot n!)$ total / $O(n)$ auxiliary
  - Output: $n!$ permutations, each of length $n$.
  - Auxiliary: `path` holds at most `n` elements, `used` is `n` booleans, and the recursion depth is exactly `n`.

---

## 4. Edge Cases & Gotchas

- [x] **Single element (`n = 1`):** The loop picks the only unused index, `path` reaches size 1, and `[[nums[0]]]` is returned ($1! = 1$).
- [x] **Size 2 (`n = 2`):** Produces both `[a, b]` and `[b, a]` — confirms that revisiting earlier positions is allowed.
- [x] **State leakage between calls:** `res.clear()`, `path.clear()`, and `used.assign(n, false)` in the public method reset everything if the same `Solution` object is reused.
- [x] **`used` restore:** `used[i] = false` must run after `path.pop_back()`; otherwise sibling branches see the index as taken and whole groups of permutations vanish.
- [x] **Empty input (not in constraints):** `path.size() == 0 == nums.size()` immediately saves an empty permutation; guard defensively if the caller can pass `[]`.
- [x] **Distinct elements guaranteed:** duplicates are not handled here; that is the exact reason Permutations II needs `sort` plus the "same value as previous, previous unused" skip.
- [x] **Do not use the `start` index:** substituting `start` for `used` would restrict the search to non-decreasing index order and return only one permutation.
- **Common pitfall:** checking `path.size() == nums.size()` **after** the loop (as in Subsets) instead of at the top of the call, which records nothing or records partial states.

---

## 5. Clean Code (Optimal Solution: Backtracking + `used`)

```cpp
class Solution {
private:
    vector<vector<int>> res;
    vector<int> path;
    vector<bool> used;

    void backtrack(vector<int>& nums) {
        if (path.size() == nums.size()) {
            res.push_back(path);
            return;
        }

        for (int i = 0; i < (int)nums.size(); ++i) {
            if (used[i]) {
                continue;
            }

            // Choose
            used[i] = true;
            path.push_back(nums[i]);

            // Explore
            backtrack(nums);

            // Undo choice
            path.pop_back();
            used[i] = false;
        }
    }

public:
    vector<vector<int>> permute(vector<int>& nums) {
        res.clear();
        path.clear();
        used.assign(nums.size(), false);

        backtrack(nums);
        return res;
    }
};
```

---

## 6. Review & Takeaways

### Visual Recursion Tree ($nums = [1, 2, 3]$)

```text
backtrack()  path = []
├── pick 1 (used = [T,_,_])  path = [1]
│   ├── pick 2 (used = [T,T,_])  path = [1,2]
│   │   └── pick 3  path = [1,2,3]  save
│   └── pick 3 (used = [T,_,T])  path = [1,3]
│       └── pick 2  path = [1,3,2]  save
├── pick 2 (used = [_,T,_])  path = [2]
│   ├── pick 1  path = [2,1]
│   │   └── pick 3  path = [2,1,3]  save
│   └── pick 3  path = [2,3]
│       └── pick 1  path = [2,3,1]  save
└── pick 3 (used = [_,_,T])  path = [3]
    ├── pick 1  path = [3,1]
    │   └── pick 2  path = [3,1,2]  save
    └── pick 2  path = [3,2]
        └── pick 1  path = [3,2,1]  save

Final Result: [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]
Total: 3! = 6 permutations
```

---

### The Architectural Mental Model

```text
                        Permutations(nums)
                               ↓
                Backtracking with used[] membership
                               ↓
                 path.size() == n ?  ── yes ──> save path, return
                               │ no
                               ↓
                 for i in every unused index
                               ↓
                    used[i] = true, path.push(nums[i])
                               ↓
                        recurse()
                               ↓
                    path.pop(), used[i] = false
                               ↓
                     try next unused index
```

* **Next Review Date:**
* **Key Takeaway:** When order matters, track membership with a `used[]` array and loop over **all** indices (not a `start` bound); restore both `path` and `used` on the way back, and every leaf of the tree is a valid permutation.
