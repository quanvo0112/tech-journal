# 0079. Word Search

- **Problem Link:** https://leetcode.com/problems/word-search/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Backtracking`
- **LeetCode Topics:** `Array` / `String` / `Backtracking` / `Matrix`
- **Core Pattern:** `DFS + Backtracking (Choose -> Explore -> Undo) on a Grid`
- **Last Practiced:** 2026-10-24
- **Proficiency Level:**
  - [ ] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - *"Given an `m x n` grid of characters `board` and a string `word`, return `true` if `word` exists in the grid."*
  - *"The word can be constructed from letters of sequentially adjacent cells (horizontally or vertically neighboring)."*
  - *"The same letter cell may not be used more than once."*
  - Constraints are small: $m, n \le 6$ and $L \le 15$, which is the signature of an **exhaustive search with pruning** rather than a polynomial algorithm.
- **Key observations:**
  - The target is a **path**, not a set: order and adjacency matter, and each cell may appear at most once in the path.
  - Trying every start cell is mandatory because the word may begin anywhere.
  - The search state is fully described by `(row, col, index_in_word)`, so a DFS can carry only the current position and the matched prefix length.
- **Core insight / invariant:**
  $$\text{At depth } idx,\ \text{the cells marked } \# \text{ are exactly the cells on the current path.}$$
  - A cell is valid to step on only if it is in bounds, matches `word[idx]`, and is not already on the current path.
  - Because the marker is placed **before** exploring and removed **after**, sibling branches always see the original board: the grid is a shared, temporarily mutated resource rather than a copied state.
- **Why naive approaches fall short:**
  - Searching each cell independently without marking would allow the path to loop and reuse the same cell, producing false positives such as `"ABCB"` on a board with a single `B`.
  - A separate `visited` matrix works but allocates and copies extra state; the board itself can serve as the visited set by overwriting the matched character.

---

## 2. Approaches & Trade-offs

1. **Approach 1 — Brute Force Path Enumeration (no pruning):**
   - From each start cell, enumerate every walk of length $L$ in 4 directions and compare with `word`.
   - *Complexity:* $O(mn \cdot 4^L)$.
   - *Drawback:* Rejects nothing early; explores paths whose current character already mismatches the target.

2. **Approach 2 — BFS over `(row, col, index, visited_set)`:**
   - Push states into a queue, carrying an explicit set of used cells (or a bitmask) per state.
   - *Drawback:* Exponential state duplication; each queued state must copy or encode its own visited set, and backtracking is not natural because the path is no longer a single mutable stack.

3. **Approach 3 — DFS + Backtracking with In-Place `'#'` Marker (Chosen Solution):**
   - `dfs(r, c, idx)`:
     1. **Bounds check** — reject `r`/`c` outside the grid.
     2. **Match check** — reject if `board[r][c] != word[idx]`.
     3. **Terminal check** — if `idx == word.size() - 1`, the whole word matched: return `true`.
     4. **Choose** — save `board[r][c]`, then set it to `'#'`.
     5. **Explore** — recurse into up, down, left, right with `idx + 1`.
     6. **Undo** — restore the saved character.
   - The driver tries `dfs(r, c, 0)` for every cell and returns `true` on the first success.
   - *Why this variant:*
     - The `'#'` marker is safe because the problem guarantees only English letters in `board`, so `'#'` can never collide with real data. It removes the need for a second `vector<vector<bool>> visited` and its per-branch copies.
     - Marking is done **after** the character match and terminal check, so a single-character word or the last character never mutates the board unnecessarily.
     - `found` is computed first and the cell restored afterwards, guaranteeing the undo runs on both success and failure — a common source of bugs when using early `return dfs(...)` after marking.
     - The member `grid` pointer and `target` string avoid re-passing the board and word through every recursive frame, keeping each call to three integers.
     - The OR-chain short-circuits: once a direction succeeds, the remaining three are not explored.

---

## 3. Complexity Analysis

Let $m$ and $n$ be the board dimensions and $L = |\text{word}|$.

- **Time Complexity:** $O(mn \cdot 3^L)$ — Explanation: there are $mn$ possible starting cells. From the first cell the search may branch in up to 4 directions, but every subsequent step can return to the previous cell only by stepping on the `'#'` marker, which is rejected, so each deeper level has at most **3** viable directions. The total work is therefore bounded by $mn \cdot 4 \cdot 3^{L-1} = O(mn \cdot 3^L)$. Character mismatches cut branches early, so the real cost is usually far below this bound.
- **Space Complexity:** $O(L)$ auxiliary — Explanation: the recursion stack holds at most $L$ frames (one per matched character). No `visited` matrix is allocated because the board is mutated in place and restored during backtracking. The grid itself is not copied.

---

## 4. Edge Cases & Gotchas

- [x] **Single-character word (`word = "A"`):** the terminal check fires before marking, so the board is never mutated and the answer is simply whether any cell equals `'A'`.
- [x] **Word longer than any possible path (`L > m * n`):** cannot be matched; the DFS simply exhausts every start and returns `false`.
- [x] **Reusing a cell (`"ABCB"`):** the second `B` is already marked `'#'` on the current path, so that branch is rejected — this is exactly the case backtracking must prevent.
- [x] **Start cell mismatch:** handled by the character check, letting all $mn$ starts be attempted uniformly without a separate pre-filter.
- [x] **Out-of-bounds neighbours:** checked before indexing the board; the bounds test must come first, not after reading `grid[r][c]`.
- [x] **Board mutation leakage:** the restore step must execute on every path, including the successful one, otherwise later start cells see a corrupted board and produce wrong answers.
- [x] **`m` / `n` initialization:** taken from `board.size()` and `board[0].size()`; the constraints guarantee a non-empty board, so `board[0]` is safe.
- [x] **No duplicate start handling needed:** unlike combination problems, each start cell is an independent path and duplicates across starts are not a concern for a boolean answer.
- **Common pitfalls during implementation:**
  - Marking the cell **before** checking the character, which can leave stale `'#'` markers on mismatching cells if the restore is skipped.
  - Returning early from inside the OR-chain without restoring, e.g. `return dfs(...)` placed before the undo step.
  - Forgetting that `idx` is an `int` while `word.size()` is `size_t`; comparing them directly can trigger signed/unsigned warnings.

---

## 5. Clean Code

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
private:
    int m, n;
    vector<vector<char>>* grid;
    string target;

    bool dfs(int r, int c, int idx) {
        if (r < 0 || r >= m || c < 0 || c >= n) {
            return false;
        }

        if ((*grid)[r][c] != target[idx]) {
            return false;
        }

        if (idx == (int)target.size() - 1) {
            return true;
        }

        // Mark the current cell as visited.
        char original = (*grid)[r][c];
        (*grid)[r][c] = '#';

        bool found =
            dfs(r - 1, c, idx + 1) ||
            dfs(r + 1, c, idx + 1) ||
            dfs(r, c - 1, idx + 1) ||
            dfs(r, c + 1, idx + 1);

        // Backtrack: restore the original character.
        (*grid)[r][c] = original;

        return found;
    }

public:
    bool exist(vector<vector<char>>& board, string word) {
        m = board.size();
        n = board[0].size();
        grid = &board;
        target = word;

        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                if (dfs(r, c, 0)) {
                    return true;
                }
            }
        }

        return false;
    }
};
```

---

## 6. Review & Takeaways

### Step-by-Step Dry Run

Board:

```text
    c0 c1 c2 c3
r0   A  B  C  E
r1   S  F  C  S
r2   A  D  E  E
```

`word = "ABCCED"` — expected `true`.

```text
Start (0,0) = 'A' == word[0]:
  mark (0,0)='#'                     board marker: (0,0)
  ├─ up    (-1,0)  out of bounds
  ├─ down  (1,0)='S' != 'B'  reject
  ├─ left  (0,-1)  out of bounds
  └─ right (0,1)='B' == word[1]
       mark (0,1)='#'                markers: (0,0) (0,1)
       └─ right? up? ... explore:
          (0,2)='C' == word[2]
            mark (0,2)='#'
            (1,2)='C' == word[3]     <- the only viable neighbour
              mark (1,2)='#'
              (2,2)='E' == word[4]
                mark (2,2)='#'
                (2,1)='D' == word[5] <- last index
                return true  ("ABCCED" matched)
```

Path: `A(0,0) -> B(0,1) -> C(0,2) -> C(1,2) -> E(2,2) -> D(2,1)`, each cell used exactly once. As the stack unwinds, every `'#'` is restored to its original letter.

```text
word = "SEE"  -> S(1,3) -> E(2,3) -> E(2,2)          => true
word = "ABCB" -> A(0,0) -> B(0,1) -> C(0,2)
                 next 'B' would require (0,1), but it is '#'
                 -> all four directions fail              => false
```

---

### The Architectural Mental Model

```text
                 for every cell (r, c):  dfs(r, c, 0)
                                ↓
        ┌───────────── dfs(r, c, idx) ─────────────┐
        │ out of bounds?          -> false          │
        │ board[r][c] != word[idx]? -> false        │
        │ idx == L - 1?            -> true           │
        │ board[r][c] = '#'        (Choose)          │
        │ recurse 4 neighbours    (Explore)          │
        │ board[r][c] = original   (Undo)            │
        └───────────────────────────────────────────┘
                                ↓
             first true short-circuits the OR-chain
```

* **Next Review Date:**
* **Key Takeaway:** For "find a path in a grid where each cell is used once", use DFS on `(row, col, index)` and mutate the board itself as the visited set — mark on choose, restore on undo — so every sibling branch starts from a clean grid.
