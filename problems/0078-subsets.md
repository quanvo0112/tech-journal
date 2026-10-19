# 0078. Subsets

- **Problem Link:** https://leetcode.com/problems/subsets/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Backtracking`
- **LeetCode Topics:** `Array` / `Backtracking` / `Bit Manipulation`
- **Core Pattern:** `Subset Enumeration / Backtracking (push -> recurse -> pop)`
- **Last Practiced:** 2026-10-19
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given an integer array `nums` of **unique** elements, return all possible subsets (the power set) in any order.
  - Constraint: $1 \le \text{nums.length} \le 10$, all elements distinct.
  - Return type is `vector<vector<int>>`.
- **Core intuition:**
  - Each element has exactly **two choices**: included or excluded. For $n$ elements there are $2^n$ combinations, including the empty set and the full array.
  - **State = valid subset:** A key realization — we do not need to reach a leaf before recording a result. The moment `backtrack()` is entered, `curr` is already a complete, valid subset. Recording at the top of every call automatically captures `[]`, every prefix, and the full set.
  - **Why not brute-force bitmask?** There is no "filter" step; every one of the $2^n$ masks is valid. The relevant design question is how to enumerate them without duplicates and without reordering (e.g. `[1,2]` vs `[2,1]`).
  - **`start` index is the deduplication invariant:** The loop only offers elements at index $\ge start$. After picking `nums[i]`, recurse with `i + 1` so only *later* elements are considered. This guarantees each element is used at most once per subset and forbids order permutations from producing duplicate subsets.
  - **Choose -> Explore -> Un-choose:** `push_back` -> recurse -> `pop_back` on a shared mutable `curr` reuses one buffer across the whole recursion tree instead of allocating a fresh vector per branch.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Cascading / Iterative Expansion ($O(2^n \cdot n)$ time, $O(2^n \cdot n)$ space):**
   - Start from `result = [[]]`. For each `num`, append `num` to a copy of every existing subset and add it back.
   - *Drawback:* Requires allocating new copies of every subset at each of the $n$ iterations; less symmetric and harder to generalize to constrained enumeration.

2. **Approach 2 — Bit Manipulation ($O(2^n \cdot n)$ time, $O(2^n \cdot n)$ space):**
   - Iterate mask from `0` to `2^n - 1`; include `nums[i]` when bit `i` is set. Each mask directly encodes one subset.
   - *Drawback:* Couples the enumeration to a fixed-length index encoding; does not naturally extend to pruning (duplicates, sum constraints, etc.) required by follow-ups like Subsets II or Combination Sum.

3. **Approach 3 — Backtracking on `curr` with `start` Index (Chosen Optimal Solution):**
   - `backtrack(start)`:
     - Save `curr` to `ans` (records the empty set at the root call).
     - For `i` from `start` to `n - 1`: `push_back(nums[i])`, `backtrack(i + 1)`, `pop_back()`.
   - *Verdict:* Optimal and idiomatic. Each node of the recursion tree emits exactly one subset, uses $O(1)$ mutations, and the `start` bound is the reusable template that extends cleanly to Subsets II, Combination Sum, and Permutations.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(n \cdot 2^n)$
  - There are $2^n$ subsets, and each is copied into `ans` in up to $O(n)$ time, giving the $n \cdot 2^n$ term.
  - The recursion tree has $2^{n+1} - 1$ nodes with $O(1)$ work per node, dominated by the copy cost.
- **Space Complexity:** $O(n \cdot 2^n)$ total / $O(n)$ auxiliary
  - Output space: stores $2^n$ subsets, each up to length $n$.
  - Auxiliary: `curr` holds at most $n$ elements; the recursion depth is at most $n+1$.

---

## 4. Edge Cases & Gotchas

- [x] **Empty subset:** The root call pushes `curr` while it is empty, correctly emitting `[]`.
- [x] **Single element (`n = 1`):** Produces exactly `[[], [nums[0]]]` ($2^1 = 2$).
- [x] **Repeated calls (state leakage):** `ans.clear()` and `curr.clear()` inside the public method guard against stale state from a previous invocation on the same `Solution` object.
- [x] **`start` boundary:** Looping with `i < nums.size()` ensures `backtrack(i + 1)` degrades to an immediate save-and-return when `i` is the last index.
- [x] **No duplicates required:** The problem guarantees distinct elements; duplicate handling is deferred to Subsets II (sort + skip).
- **Common pitfall:** Recording the result *after* the loop instead of at the top of the call would drop the empty set and every intermediate prefix.

---

## 5. Clean Code (Optimal Solution: Backtracking)

```cpp
class Solution {
private:
    vector<vector<int>> ans;
    vector<int> curr;

    void backtrack(int start, vector<int>& nums) {
        // Every entry into backtrack() represents a valid subset.
        ans.push_back(curr);

        for (int i = start; i < (int)nums.size(); ++i) {
            curr.push_back(nums[i]);

            backtrack(i + 1, nums);

            curr.pop_back();  // Backtrack
        }
    }

public:
    vector<vector<int>> subsets(vector<int>& nums) {
        ans.clear();
        curr.clear();

        backtrack(0, nums);

        return ans;
    }
};
```

---

## 6. Review & Takeaways

### Visual Recursion Tree ($nums = [1, 2, 3]$)

```text
backtrack(0) save [] 
├── pick 1: backtrack(1) save [1]
│   ├── pick 2: backtrack(2) save [1,2]
│   │   └── pick 3: backtrack(3) save [1,2,3]
│   └── pick 3: backtrack(3) save [1,3]
├── pick 2: backtrack(2) save [2]
│   └── pick 3: backtrack(3) save [2,3]
└── pick 3: backtrack(3) save [3]

Final Result: [], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]
Total: 2^3 = 8 subsets
```

---

### The Architectural Mental Model

```text
                        Subsets(nums)
                              ↓
                     Backtracking (start)
                              ↓
                   Save curr at every node
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
        push nums[i]                     (next i)
              ↓
        recurse (i + 1)
              ↓
        pop_back()  ──> try next sibling
```

* **Next Review Date:**
* **Key Takeaway:** In subset enumeration, record `curr` at the top of every recursive call and pick only from index `i + 1` onward; this yields all $2^n$ subsets exactly once via the `push -> recurse -> pop` template.
