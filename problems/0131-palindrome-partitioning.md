# 0131. Palindrome Partitioning

- **Problem Link:** https://leetcode.com/problems/palindrome-partitioning/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Backtracking`
- **LeetCode Topics:** `String` / `Dynamic Programming` / `Backtracking`
- **Core Pattern:** `Interval DP Precomputation (Palindrome Table) + Backtracking Partition Enumeration`
- **Last Practiced:** 2026-10-25
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Partition `s` such that **every substring** of the partition is a palindrome, and return **all possible** palindrome partitionings.
  - Return type is `vector<vector<string>>` — an enumeration problem, not an optimization problem.
  - Constraint: $1 \le s.length \le 16$, which allows exponential enumeration.
- **Core intuition:**
  - A partition is fully determined by where the cuts go. With $n$ characters there are at most $2^{n-1}$ cut patterns, so the output itself can be exponential (e.g. `"aaaa..."` yields a Fibonacci-like / $2^{n-1}$ blowup).
  - **Backtracking is the natural fit:** at position `start`, try every `end >= start`; if `s[start..end]` is a palindrome, commit it as the next segment and recurse from `end + 1`. When `start == n`, `path` is one complete valid partition.
  - **Why the naive version wastes work:** testing `s[start..end]` with a two-pointer scan costs $O(n)$ per candidate, and the same substrings are re-validated across many sibling branches. The predicate is queried far more often than $n^2$ times.
  - **The palindrome predicate has optimal substructure:**
    $$isPal[i][j] = (s[i] == s[j]) \land \bigl(j - i \le 2 \;\lor\; isPal[i+1][j-1]\bigr)$$
    - A substring is a palindrome exactly when its two ends match and the interior is a palindrome.
    - Lengths 1, 2, and 3 need no interior check at all — that is precisely the `j - i <= 2` guard.
  - **Fill order matters:** iterate `i` from `n - 1` down to `0` (and `j` upward from `i`) so that `isPal[i+1][j-1]` — a strictly shorter interval — is already known when it is read.
  - **The invariant:** after preprocessing, `isPal[start][end]` is an $O(1)$ oracle. Backtracking then only pays for the choices it actually makes, so the combinatorial cost is not multiplied by substring re-scanning.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Enumerate cut bitmasks, then validate ($O(n \cdot 2^n)$ time, $O(n)$ auxiliary):**
   - There are $2^{n-1}$ ways to place cuts between adjacent characters. For each mask, split the string and check every segment with a two-pointer scan.
   - *Drawback:* Builds every candidate partition first — including the overwhelmingly many invalid ones — and allocates temporary strings for segments that are discarded. No pruning happens during construction.

2. **Approach 2 — Backtracking with on-the-fly palindrome checks ($O(n^2 \cdot 2^n)$ worst case, $O(n)$ auxiliary):**
   - Same `start`/`end` recursion, but validate each candidate substring by expanding or by two pointers.
   - *Drawback:* Each check is $O(n)$, so the exponential search is multiplied by a linear scan. Worse, identical `(start, end)` pairs are re-checked along different branches, so the cost is paid repeatedly with no reuse.

3. **Approach 3 — Interval DP palindrome table + Backtracking (Chosen Optimal Solution):**
   - **Step 1 (preprocess):** build `isPal[n][n]` with the recurrence above, filling `i` from `n - 1` down to `0`.
   - **Step 2 (enumerate):** `backtrack(start)`:
     - If `start == n`: `res.push_back(path)` and return.
     - For `end` from `start` to `n - 1`:
       - Skip when `!isPal[start][end]`.
       - `path.push_back(s.substr(start, end - start + 1))`, `backtrack(end + 1)`, `path.pop_back()`.
   - *Verdict:* Optimal and idiomatic. Every palindrome predicate becomes an $O(1)$ table lookup, so the exponential work is spent solely on generating real partitions. The `start`/`end` shape is the same partition skeleton reusable for Word Break-style enumeration, and the DP table is decoupled from the search.
   - **Why `vector<bool>`:** the table only ever stores truth values, so `vector<bool>` bit-packs $n^2$ flags into roughly $n^2/8$ bytes. Reading `isPal[i][j]` through the proxy reference is fine here because no address is taken.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(n^2 + n \cdot 2^n)$
  - Preprocessing fills an $n \times n$ triangular table in $O(n^2)$.
  - There are at most $2^{n-1}$ cut patterns. Each valid leaf copies a partition of up to $n$ strings of total length $n$ into `res`, costing up to $O(n)$ — hence the output-sensitive $O(n \cdot 2^n)$ term.
  - Palindrome testing contributes $O(1)$ per attempted segment, which is absorbed by the enumeration bound. This is optimal in the output-sensitive sense: every character of every emitted partition must be written at least once.
- **Space Complexity:** $O(n^2)$ auxiliary
  - `isPal` is an $n \times n$ boolean table (bit-packed by `vector<bool>`).
  - `path` holds at most $n$ substrings, and the recursion stack has depth at most $n$.
  - `res` stores the output itself and is not counted as auxiliary space.

---

## 4. Edge Cases & Gotchas

- [x] **Single character (`n = 1`):** `start == 0`, only `end = 0` matches, so the result is `[["a"]]`.
- [x] **Whole string is already a palindrome:** the one-segment partition is emitted, alongside every finer partition.
- [x] **Repeated characters (`"aaa"`):** yields `["a","a","a"]`, `["a","aa"]`, `["aa","a"]`, `["aaa"]` — the exponential output is genuine, not duplicated work.
- [x] **No empty substrings:** `end` starts at `start`, so the length is always `end - start + 1 >= 1`.
- [x] **Base case placement:** only `start == n` is complete; saving at the top of `backtrack()` would emit partial prefixes as if they were full partitions.
- [x] **DP fill order:** iterating `i` left-to-right would read `isPal[i+1][j-1]` before it is computed and silently produce wrong flags. Loop `i` from `n - 1` down to `0`.
- [x] **Short-circuit safety:** `j - i <= 2 || isPal[i + 1][j - 1]` relies on `||` short-circuiting so the inner index is never accessed for length 1–3 substrings.
- [x] **State restoration:** forgetting `path.pop_back()` leaks the chosen segment into sibling branches and corrupts every later partition.
- [x] **Repeated calls on one object:** `res.clear()` and `path.clear()` inside `partition()` keep the `Solution` reusable; `isPal.assign(...)` also resets the table size.

---

## 5. Clean Code (Optimal Solution: DP + Backtracking)

```cpp
class Solution {
private:
    vector<vector<string>> res;
    vector<string> path;
    vector<vector<bool>> isPal;
    int n;

    void backtrack(const string& s, int start) {
        if (start == n) {
            res.push_back(path);
            return;
        }

        for (int end = start; end < n; ++end) {
            if (!isPal[start][end]) {
                continue;
            }

            // Choose
            path.push_back(s.substr(start, end - start + 1));

            // Explore the remaining string
            backtrack(s, end + 1);

            // Undo choice
            path.pop_back();
        }
    }

public:
    vector<vector<string>> partition(string s) {
        n = s.size();
        res.clear();
        path.clear();

        isPal.assign(n, vector<bool>(n, false));

        // Precompute whether each substring is a palindrome
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i; j < n; ++j) {
                isPal[i][j] = (s[i] == s[j]) &&
                              (j - i <= 2 || isPal[i + 1][j - 1]);
            }
        }

        backtrack(s, 0);
        return res;
    }
};
```

---

## 6. Review & Takeaways

### Visual Dry Run (`s = "aab"`)

**Step 1 — Precomputed palindrome table**

```text
        j=0     j=1     j=2
      +-------+-------+-------+
 i=0  |  T    |  T    |  F    |   "a", "aa", "aab"(x)
      +-------+-------+-------+
 i=1  |       |  T    |  F    |   "a", "ab"(x)
      +-------+-------+-------+
 i=2  |       |       |  T    |   "b"
      +-------+-------+-------+
```

**Step 2 — Backtracking over segment boundaries**

```text
backtrack(0)
├── end=0: isPal[0][0] T -> path ["a"] -> backtrack(1)
│   ├── end=1: isPal[1][1] T -> path ["a","a"] -> backtrack(2)
│   │   └── end=2: isPal[2][2] T -> path ["a","a","b"] -> backtrack(3) [SAVE]
│   └── end=2: isPal[1][2] F -> skip
├── end=1: isPal[0][1] T -> path ["aa"] -> backtrack(2)
│   └── end=2: isPal[2][2] T -> path ["aa","b"] -> backtrack(3) [SAVE]
└── end=2: isPal[0][2] F -> skip

Final Result: [["a","a","b"], ["aa","b"]]
```

---

### The Architectural Mental Model

```text
                    Palindrome Partitioning (s)
                                ↓
                 Interval DP: fill isPal[i][j]
                 (i from n-1 down to 0)
                                ↓
                    backtrack(start = 0)
                                ↓
                 ┌──────────────┴──────────────┐
                 ↓                             ↓
        start == n ?                    for end in [start, n)
        YES -> save path                        ↓
                                        isPal[start][end] ?
                                          /            \
                                        NO              YES
                                        ↓                ↓
                                      skip        push s[start..end]
                                                         ↓
                                              backtrack(end + 1)
                                                         ↓
                                                   pop_back()
```

* **Next Review Date:**
* **Key Takeaway:** Precompute the palindrome predicate with an interval DP so each `s[start..end]` check is $O(1)$, then backtrack over cut boundaries by choosing a palindromic prefix and recursing on the suffix from `end + 1`.
