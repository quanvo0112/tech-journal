# 0051. N-Queens

- **Problem Link:** https://leetcode.com/problems/n-queens/
- **Difficulty:** `Hard`
- **NeetCode Category:** `Backtracking`
- **LeetCode Topics:** `Array` / `Backtracking`
- **Core Pattern:** `Constrained Backtracking with O(1) Conflict Sets (Column + Two Diagonals)`
- **Last Practiced:** 2026-10-27
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Place `n` queens on an `n x n` board so that **no two queens attack each other**, and return **all distinct boards**.
  - Attack lines are rows, columns, and both diagonals; the answer must be materialized as an explicit board (`vector<string>`), so membership is decided by geometry, not by values.
  - Constraint is tiny: $1 \le n \le 9$. Exhaustive search is intended — but only if each placement check is cheap.
  - `n = 2` and `n = 3` have **zero** valid arrangements, which is a useful mental test for pruning correctness.
- **Core intuition:**
  - Placing exactly **one queen per row** is a free structural decision: it removes row conflicts by construction, so the search level is simply `row`. Only three conflict families remain — column, `/` diagonal, `\` diagonal.
  - Each conflict family is a **set of lines**, and each line has a closed-form integer label:
    - Column: `col`.
    - Anti-diagonal (`/`, direction ↗/↙): every cell has the same `row + col`, ranging over `[0, 2n - 2]`.
    - Main diagonal (`\`, direction ↘/↖): every cell has the same `row - col`, ranging over `[-(n-1), n-1]` — shift by `n - 1` to get a valid index `row - col + n - 1`.
  - **The invariant:** three boolean arrays tell whether a line is occupied. A candidate `(row, col)` is legal **iff** all three corresponding flags are `false`. That is an $O(1)$ oracle instead of an $O(n)$ board scan.
  - **Choose -> Explore -> Undo:** write `'Q'`, set three flags, recurse to `row + 1`; then restore `'.'` and clear the three flags. The board is a single mutable buffer, never copied during search.
  - Reaching `row == n` means `n` queens occupy `n` rows with no shared line — the board is a complete valid solution and must be **copied** into `res`.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Enumerate all $n^n$ assignments, then validate each board ($O(n^n \cdot n^2)$ time, $O(n^2)$ space):**
   - Generate every possible column for every row (or even every cell subset of size $n$), then run a full board validator that checks rows, columns, and diagonals.
   - *Drawback:* Explores placements that are already invalid, and pays $O(n^2)$ validation per leaf. Completely wasteful at $n = 9$.
2. **Approach 2 — Backtracking with an `isSafe()` board scan ($O(n! \cdot n)$ time, $O(n^2)$ space):**
   - For each `(row, col)`, walk the column upward and both diagonal directions, checking for an existing `'Q'`.
   - *Drawback:* Each candidate check is $O(n)$, so the constant factor multiplies the entire search tree. Correct and much better than Approach 1, but it re-derives the same geometry thousands of times.
3. **Approach 3 — Backtracking with boolean conflict arrays (Chosen Optimal Solution):**
   - `cols[n]`, `diag1[2n - 1]` ($row + col$), `diag2[2n - 1]` ($row - col + n - 1$).
   - `backtrack(row)`:
     - If `row == n`: `res.push_back(board)` and return.
     - For `col` in `[0, n)`: compute `d1 = row + col`, `d2 = row - col + n - 1`; skip if any of the three flags is set.
     - Otherwise mark `'Q'` + three flags, recurse on `row + 1`, then unmark all four.
   - Public method: `res.clear()`, build the all-`'.'` board and the three arrays sized `n` / `2n - 1` / `2n - 1`, then `backtrack(0)`.
   - *Verdict:* The line labels replace geometry with arithmetic, so each candidate is judged in $O(1)$. Array indexing beats `unordered_set` here: no hashing, no allocation, and the index range is known and dense.
   - **Why no row array:** the recursion fixes exactly one queen per row, so two queens can never share a row. Adding a `rows[]` flag would be dead state.
   - **Why `diag2` needs the `+ n - 1` shift:** `row - col` is negative for half the board; the shift maps $[-(n-1), n-1]$ onto $[0, 2n - 2]$ without changing which cells collide.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(n! \cdot n^2)$ output-inclusive; the search itself is commonly stated as $O(n! \cdot n)$ / loosely $O(n \cdot n!)$
  - Row 0 has `n` choices, row 1 at most `n - 1` after the column conflict, row 2 at most `n - 2`, and so on — pruning collapses the naive $n^n$ tree toward $n!$ partial assignments.
  - Each node performs $O(1)$ flag work (the three array reads). The expensive step happens only at a solution: copying an $n \times n$ board, i.e. $O(n^2)$ per solution.
  - If there are $S$ solutions (for $n = 9$, $S = 352$), the total is $O(\text{search nodes} + S \cdot n^2)$; the $S \cdot n^2$ term is the mandatory output cost.
- **Space Complexity:** $O(n^2)$ auxiliary (excluding `res`)
  - `board` is $n$ strings of length $n$ — $O(n^2)$.
  - `cols` is $n$, `diag1` and `diag2` are $2n - 1$ each — $O(n)$ combined, dominated by the board.
  - Recursion depth is exactly `n`, so the call stack is $O(n)$.
  - `res` holds $S$ boards of $n^2$ characters and is output, not auxiliary.

---

## 4. Edge Cases & Gotchas

- [x] **`n = 1`:** `backtrack(0)` immediately places the only queen, `row == 1 == n`, returns one board with a single `'Q'`.
- [x] **`n = 2` / `n = 3`:** no arrangement exists; the recursion exhausts every branch and returns `[]`. Do not special-case it — correct pruning must produce the empty vector naturally.
- [x] **Negative diagonal index:** `row - col` can be as low as $-(n-1)$. Omitting the `+ n - 1` shift is an out-of-bounds / undefined-behavior bug.
- [x] **Array sizing:** both diagonal arrays need exactly `2 * n - 1` slots. `n` slots silently truncate the range for $n > 1$.
- [x] **Board initialization:** `board.assign(n, string(n, '.'))` — creating `n` *distinct* rows. A single `string` reused by reference would corrupt every row at once.
- [x] **Copy at the leaf:** `res.push_back(board)` must copy the board, because the same `board` object keeps mutating during later branches.
- [x] **Full undo:** restore `board[row][col] = '.'` and reset **all three** flags. Forgetting one leaves a phantom line that suppresses every later candidate on it.
- [x] **`d1` / `d2` consistency between check and mark:** compute both indices once per loop iteration and reuse them for the guard, the set, and the clear. Recomputing with a different formula silently desynchronizes the state.
- [x] **Repeated calls on one object:** `res.clear()` and re-`assign` all arrays in the public method, preventing leaked state between invocations.
- **Common pitfall:** carrying a `rows[]` array or a `used[]` array from the permutation template. The row dimension is the recursion level itself, so extra row bookkeeping is redundant.

---

## 5. Clean Code (Optimal Solution: Backtracking + Boolean Arrays)

```cpp
class Solution {
private:
    vector<vector<string>> res;
    vector<string> board;
    vector<bool> cols;
    vector<bool> diag1;
    vector<bool> diag2;
    int n;

    void backtrack(int row) {
        if (row == n) {
            res.push_back(board);
            return;
        }

        for (int col = 0; col < n; ++col) {
            int d1 = row + col;
            int d2 = row - col + n - 1;

            if (cols[col] || diag1[d1] || diag2[d2]) {
                continue;
            }

            // Choose: place a queen
            board[row][col] = 'Q';
            cols[col] = true;
            diag1[d1] = true;
            diag2[d2] = true;

            // Explore: place a queen in the next row
            backtrack(row + 1);

            // Backtrack: remove the queen
            board[row][col] = '.';
            cols[col] = false;
            diag1[d1] = false;
            diag2[d2] = false;
        }
    }

public:
    vector<vector<string>> solveNQueens(int n) {
        this->n = n;

        res.clear();
        board.assign(n, string(n, '.'));
        cols.assign(n, false);
        diag1.assign(2 * n - 1, false);
        diag2.assign(2 * n - 1, false);

        backtrack(0);
        return res;
    }
};
```

---

## 6. Review & Takeaways

### Visual Dry-Run (`n = 4`, tracing the first solution)

```text
Line labels:  d1 = row + col  (anti-diagonal "/")
              d2 = row - col + 3  (main diagonal "\")

row 0, col 0 -> place Q(0,0)            d1=0 d2=3
  row 1, col 0 -> SKIP (cols)
  row 1, col 1 -> SKIP (d2 = 3, same as Q(0,0))
  row 1, col 2 -> place Q(1,2)          d1=3 d2=2
    row 2, col 0 -> SKIP (cols, Q(0,0))
    row 2, col 1 -> SKIP (d1 = 3, same as Q(1,2))
    row 2, col 2 -> SKIP (cols, Q(1,2))
    row 2, col 3 -> SKIP (d2 = 2, same as Q(1,2))
    -> dead end, undo Q(1,2)
  row 1, col 3 -> place Q(1,3)          d1=4 d2=1
    row 2, col 0 -> place Q(2,0)        d1=2 d2=5
      row 3, col 0 -> SKIP (cols)
      row 3, col 1 -> SKIP (d1 = 4, same as Q(1,3))
      row 3, col 2 -> place Q(3,2)      d1=5 d2=4
        row 4 == n -> SAVE board
```

Resulting board for that branch:

```text
col:   0 1 2 3
row 0: . Q . .
row 1: . . . Q
row 2: Q . . .
row 3: . . Q .
```

For `n = 4` the two valid solutions are:

```text
Solution 1        Solution 2
. Q . .           . . Q .
. . . Q           Q . . .
Q . . .           . . . Q
. . Q .           . Q . .
```

---

### The Architectural Mental Model

```text
                 solveNQueens(n)
                        ↓
        board = n x n of '.', all flags = false
                        ↓
                 backtrack(row = 0)
                        ↓
        row == n ? ──YES──> copy board into res
                        │
                       NO
                        ↓
        for col = 0 .. n-1:
            d1 = row + col, d2 = row - col + n - 1
                        ↓
        cols[col] || diag1[d1] || diag2[d2] ? ──YES──> skip
                        │NO
                        ↓
        board[row][col] = 'Q', mark 3 flags
                        ↓
                backtrack(row + 1)
                        ↓
        board[row][col] = '.', clear 3 flags
```

* **Next Review Date:**
* **Key Takeaway:** Turn each attack line into an integer label (`col`, `row + col`, `row - col + n - 1`), mark those lines in three boolean arrays, and backtrack one row at a time — every placement check becomes $O(1)$, and the only real cost left is copying each finished $n \times n$ board.
