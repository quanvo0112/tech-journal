# 0678. Valid Parenthesis String

- **Problem Link:** https://leetcode.com/problems/valid-parenthesis-string/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Greedy`
- **LeetCode Topics:** `String` / `Dynamic Programming` / `Stack` / `Greedy`
- **Core Pattern:** `Greedy Range Tracking / Open Bracket Balance Interval [low, high]`
- **Last Practiced:** 2026-10-04
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given a string `s` containing only three types of characters: `'('`, `')'`, and `'*'`.
  - Return `true` if `s` is valid, otherwise return `false`.
  - `'*'` can be treated as a single right parenthesis `')'`, a single left parenthesis `'('`, or an empty string `""`.
  - An empty string is also valid.
  - Constraints: Length of `s` in $[1, 100]$.
- **The Core Difficulty (Uncertainty of Wildcard `*`):**
  - In classical parenthesis validation (**LeetCode 0020 Valid Parentheses**), the count of unclosed open parentheses is a single deterministic integer `balance`.
  - Here, each `'*'` branches into 3 potential futures:
    - Treating `'*'` as `'('` increases balance by $1$.
    - Treating `'*'` as `')'` decreases balance by $1$.
    - Treating `'*'` as `""` leaves balance unchanged.
  - Making an immediate hard decision for each `'*'` is impossible without looking ahead.
- **Why Naive Search Explodes:**
  - Recursion or backtracking branching $3$ ways per `'*'` leads to $O(3^N)$ exponential worst-case time complexity.
- **The Greedy Insight (Interval of Feasible Balances):**
  - Because each wildcard can shift the net balance by $\pm 1$ or $0$, the set of all valid unclosed open parenthesis counts at any prefix always forms a continuous range of integers:
    $$\text{balance} \in [\text{low}, \text{high}]$$
    - `low`: The **minimum possible** number of unclosed `'('` open brackets (optimistically treating wildcards as `')'` or `""`).
    - `high`: The **maximum possible** number of unclosed `'('` open brackets (optimistically treating wildcards as `'('`).
  - Instead of exploring branching paths, we simply update the bounds `[low, high]` greedily in $O(1)$ space.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Backtracking / Exhaustive Search ($O(3^N)$ time, $O(N)$ space):**
   - Recurse through all three branches for every `'*'`.
   - *Drawback:* Catastrophic exponential runtime; times out quickly for longer strings.

2. **Approach 2 — 2D Dynamic Programming / Memoization ($O(N^2)$ time, $O(N^2)$ space):**
   - Let `dp(i, balance)` be a boolean function checking if suffix `s[i:]` can be valid given current open balance.
   - *Drawback:* Correct and polynomial, but requires $O(N^2)$ matrix memory and quadratic runtime.

3. **Approach 3 — Two Stacks ($O(N)$ time, $O(N)$ space):**
   - Use one stack storing indices of `'('` and another stack storing indices of `'*'`.
   - Pop and match characters from right to left while ensuring index of `'*'` is greater than index of `'('`.
   - *Drawback:* Requires auxiliary heap/stack data structures and two-phase cleanup.

4. **Approach 4 — Greedy Range Tracking `[low, high]` (Chosen Optimal Solution):**
   - Initialize `low = 0`, `high = 0`.
   - Traverse each character `c` in `s`:
     - If `c == '('`: `++low; ++high;` (must open bracket).
     - If `c == ')'`: `low = max(0, low - 1); --high;` (must close bracket).
     - If `c == '*'`: `low = max(0, low - 1); ++high;` (`*` can act as `')'` to decrease or `'('` to increase).
   - **Early Invariant Check:** If `high < 0`, even the most optimistic choice (all `'*'` acting as `'('`) has too many `')'`. Return `false` immediately.
   - **Final Condition:** At string termination, return `low == 0`. If `low == 0`, a balance of exactly $0$ lies within the achievable interval $[0, \text{high}]$, meaning a valid full pairing exists.
   - *Verdict:* Optimal $O(N)$ time and strictly $O(1)$ auxiliary space.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(N)$
  - Where $N$ is the length of string `s`.
  - We iterate through `s` exactly once, performing constant $O(1)$ arithmetic operations and bounds clamping per character.
- **Space Complexity:** $O(1)$
  - Only two integer scalar variables (`low` and `high`) are maintained. Zero auxiliary data structures or recursion call stacks allocated.

---

## 4. Edge Cases & Gotchas

- [x] **Excess `')'` Early Failure (`high < 0`):** E.g., `s = "())"`. At index 2, `high` drops to `-1`. No assignment of earlier characters can rescue the string. Returns `false` immediately.
- [x] **Clamping `low` to $0$ (`low = max(0, low - 1)`):** A prefix can never have a negative balance of open brackets. If `low` would drop below $0$, it merely means we choose `'*'` to be an empty string `""` or we matched an earlier open bracket, so the minimum requirement of open brackets to close remains $0$.
- [x] **Unmatched Open Parentheses at End (`low > 0`):** E.g., `s = "(*"`. `low = max(0, 1 - 1) = 0`, `high = 2`. Since `low == 0`, `*` can be empty and `(` remains unmatched? Wait, for `s = "(*"`:
  - After `(`: `[1, 1]`
  - After `*`: `low = max(0, 0) = 0`, `high = 2`.
  - Is `(*` valid? If `* = ')'`, `"()"` is valid! Thus `low == 0` correctly returns `true`.
  - Contrast with `s = "(": After `(`, `[1, 1]`. End: `low = 1 != 0` -> returns `false`.
- [x] **Wildcards Cannot Form Negative Balances:** For `s = "*)"`, after `*`, `[0, 1]`. After `)`, `low = max(0, -1) = 0`, `high = 0`. At end, `low == 0` -> returns `true` (by choosing `*` as `'('`, forming `"()"`).

---

## 5. Clean Code (Optimal Solution: Greedy Balance Range)

```cpp
class Solution {
public:
    bool checkValidString(string s) {
        int low = 0;
        int high = 0;

        for (char c : s) {
            if (c == '(') {
                ++low;
                ++high;
            } 
            else if (c == ')') {
                low = max(0, low - 1);
                --high;
            } 
            else { // '*'
                // '*' can act as ')', '(', or empty string ""
                low = max(0, low - 1);
                ++high;
            }

            // Even in the most optimistic scenario, we have too many ')'
            if (high < 0) {
                return false;
            }
        }

        // low == 0 indicates 0 is within the feasible balance range [low, high]
        return low == 0;
    }
};
```

---

## 6. Review & Takeaways

### Visual Interval Evolution Dry-Run

#### Example 1: `s = "(*)"`

```text
Char     Action                                  [low, high]
------------------------------------------------------------
Start    Initial State                           [0, 0]
'('      Must open: low+1, high+1                [1, 1]
'*'      Wildcard: low-1 (min 0), high+1         [0, 2]
')'      Must close: low-1 (min 0), high-1       [0, 1]
------------------------------------------------------------
End:     low == 0 (0 in [0, 1]) -> VALID (true)
         (Corresponds to choosing '*' as empty "")
```

#### Example 2: `s = "(*))"`

```text
Char     Action                                  [low, high]
------------------------------------------------------------
Start    Initial State                           [0, 0]
'('      Must open: low+1, high+1                [1, 1]
'*'      Wildcard: low-1 (min 0), high+1         [0, 2]
')'      Must close: low-1 (min 0), high-1       [0, 1]
')'      Must close: low-1 (min 0), high-1       [0, 0]
------------------------------------------------------------
End:     low == 0 (0 in [0, 0]) -> VALID (true)
         (Corresponds to choosing '*' as '(' to form "(())")
```

#### Example 3: `s = "())"` (Early Failure)

```text
Char     Action                                  [low, high]
------------------------------------------------------------
'('      Must open: [1, 1]                       [1, 1]
')'      Must close: [0, 0]                      [0, 0]
')'      Must close: low=0, high = -1            [0, -1]
------------------------------------------------------------
Check:   high < 0 -> INVALID (returns false immediately)
```

---

### The Architectural Mental Model

```text
                       Valid Parenthesis String
                                  ↓
                        Track [low, high] Range
                                  ↓
                       For each character c in s:
                       ┌─────────┬─────────┬─────────┐
                       │  '('    │   ')'   │   '*'   │
                       ├─────────┼─────────┼─────────┤
                       │ low++   │ low-1   │ low-1   │
                       │ high++  │ high-1  │ high++  │
                       └─────────┴─────────┴─────────┘
                                  ↓
                         low = max(0, low)
                                  ↓
                             high < 0 ?
                              /      \
                        YES  /        \  NO
                            ↓          ↓
                       return false   Continue loop
                                  ↓
                             End of string:
                             return low == 0
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** When wildcards create non-deterministic choices, do not branch with backtracking; represent the state as a continuous valid interval `[low, high]` and greedily advance both boundaries in $\mathcal{O}(1)$ space.
