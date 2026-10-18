# 0212. Word Search II

- **Problem Link:** https://leetcode.com/problems/word-search-ii/
- **Difficulty:** `Hard`
- **NeetCode Category:** `Tries`
- **LeetCode Topics:** `Array` / `String` / `Backtracking` / `Trie` / `Matrix`
- **Core Pattern:** `Trie + DFS Backtracking + Trie Branch Pruning`
- **Last Practiced:**
- **Proficiency Level:**
  - [ ] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - *"Given an `m x n` board of characters and a list of strings `words`, return all words on the board."*
  - *"Each word must be constructed from letters of sequentially adjacent cells... The same letter cell may not be used more than once in a word."*
  - Multiple target words must be found inside a single grid.
- **Why naive word-by-word search falls short:**
  - Running a separate DFS for each word costs $O(W \cdot mn \cdot 3^L)$, where $W$ is the number of words and $L$ the max word length.
  - Words that share prefixes (`"oath"`, `"oat"`, `"oatmeal"`) re-walk the *same* board paths over and over — pure duplicated work.
  - You cannot abandon a search early: even if no word starts with the current path, the isolated DFS still explores it to full depth.
- **Core intuition:**
  - Build **one Trie** containing every word, then explore the board **once**, carrying the current Trie node along the path.
  - The Trie makes the board walk *prefix-aware*: at each cell you either continue down a valid branch or stop immediately.
  - Fundamental invariant:
    $$\text{The current board path is worth exploring} \iff \text{it traces a valid Trie branch from the root.}$$
  - A node can represent both a prefix and a completed word, so **store the full word string** at the terminal node to emit it directly when reached.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Independent DFS per Word (Naive):**
   - For each word, run the classic `0079` grid backtracking search.
   - *Verdict:* Correct but re scans the board $W$ times and re-walks shared prefixes; too slow when `words` is large.

2. **Approach 2 — Hash Set of the Word List + DFS:**
   - Store all words in a set and DFS every starting cell to max length, testing the set at each step.
   - *Trade-off:* Removes per-word restarts but still explores all $3^L$ paths with no prefix pruning, because a set cannot tell you "this prefix leads nowhere".

3. **Approach 3 — Build One Trie Once, Explore Board Once (Optimal & Implemented):**
   - Build a Trie from all `words`. Each terminal node stores the **complete word** (`string word`) instead of a boolean `isEnd`, so found words can be pushed without rebuilding the string.
   - At every cell, index the Trie by `board[r][c] - 'a'`:
     - If the child does **not** exist -> the current path is not a prefix of any remaining word -> return immediately (**prefix pruning**).
     - If the node holds a word -> record it, then call `node->word.clear()` to prevent duplicates when the same word is reachable along another board path.
   - Mark the cell with `'#'`, recurse into the 4 neighbours (bounds + `!= '#'`), then restore the cell (**backtracking**).
   - **Branch pruning:** after unwinding, if `node->isEmpty()` (no word and no surviving children), delete the branch with `parent->children[idx].reset()`. This permanently shrinks the Trie, so later DFS from other cells do not re-enter dead subtrees.
   - *Why this variant:* `array<unique_ptr<TrieNode>, 26>` gives $O(1)$ indexed descent and automatic subtree cleanup via `unique_ptr`; storing the word at the node + `clear()` gives $O(1)$ duplicate suppression; `isEmpty()` + `reset()` turns the Trie into a self-pruning structure that accelerates the remaining search. Same fixed-array Trie foundation as `0208` / `0211`, extended with grid traversal and pruning.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(S + mn \cdot 3^L)$ — Explanation: building the Trie costs $O(S)$, the total length of all words. The board DFS has an upper bound of $O(mn \cdot 3^L)$ (each of the $mn$ starting cells branches into at most 3 directions after the first step, to depth $L$), with $O(1)$ work per step for the array-index descent. In practice both the prefix cut-off and the branch deletion prune aggressively, so far fewer than $3^L$ paths are visited.
- **Space Complexity:** $O(S + L)$ excluding the output list — Explanation: the Trie holds at most $O(S)$ nodes, and the recursion stack holds $O(L)$ frames. The board is mutated in place (`'#'` marker) rather than copied, so no extra grid allocation is needed.

---

## 4. Edge Cases & Gotchas

- [ ] **Word longer than the largest possible board path:** never matched; bounded naturally by the recursion depth and the Trie branch ending.
- [ ] **Duplicate words in `words`:** a single Trie node stores one copy; `node->word.clear()` after emitting ensures it is appended once even if reachable via many board paths.
- [ ] **Self-overlapping / shared-letter words (`"oa"`, `"oath"`, `"oatmeal"`):** shared Trie prefixes are walked once per board start, not $W$ times.
- [ ] **Single-cell word (`words = ["a"]`, board `[["a"]]`):** must still mark, emit, restore, and prune the now-empty branch correctly.
- [ ] **Word that is a prefix of another (`"oat"` vs `"oatmeal"`):** emitting `"oat"` and clearing its `word` must NOT delete the node while it still has children for `"oatmeal"`; the `isEmpty()` check must see the surviving children and keep the branch.
- [ ] **Revisiting a cell in the same word:** prevented by the in-place `'#'` marker, which the bounds check rejects; restoration after recursion is mandatory.
- Common pitfalls during implementation:
  - Using a single `bool isEnd` here means reconstructing the word during DFS; storing the string at the node avoids that.
  - Forgetting the `'#'` restore, which corrupts subsequent DFS starts.
  - Pruning with `reset()` while the node still has a valid continuation (must check `isEmpty()` first, and only ever reset the **parent's** child slot).
  - Duplicate results when the same word is found along multiple board paths — solved by clearing `node->word` at first find.

---

## 5. Clean Code

```cpp
#include <array>
#include <memory>
#include <string>
#include <vector>

using namespace std;

class Solution {
private:
    struct TrieNode {
        array<unique_ptr<TrieNode>, 26> children{};
        string word;

        bool isEmpty() const {
            if (!word.empty()) {
                return false;
            }

            for (const auto& child : children) {
                if (child) {
                    return false;
                }
            }

            return true;
        }
    };

    TrieNode root;
    vector<string> ans;
    int m, n;

    void dfs(vector<vector<char>>& board, int r, int c,
             TrieNode* parent) {
        int idx = board[r][c] - 'a';

        // No matching prefix in the Trie.
        if (!parent->children[idx]) {
            return;
        }

        TrieNode* node = parent->children[idx].get();

        // Found a complete word.
        if (!node->word.empty()) {
            ans.push_back(node->word);
            node->word.clear();  // Avoid adding duplicates.
        }

        // Mark the current cell as used in this path.
        char ch = board[r][c];
        board[r][c] = '#';

        const int dr[4] = {-1, 1, 0, 0};
        const int dc[4] = {0, 0, -1, 1};

        for (int d = 0; d < 4; ++d) {
            int nr = r + dr[d];
            int nc = c + dc[d];

            if (nr >= 0 && nr < m &&
                nc >= 0 && nc < n &&
                board[nr][nc] != '#') {
                dfs(board, nr, nc, node);
            }
        }

        // Backtrack: restore the original cell.
        board[r][c] = ch;

        // Remove the branch if no word remains to be found.
        if (node->isEmpty()) {
            parent->children[idx].reset();
        }
    }

public:
    vector<string> findWords(vector<vector<char>>& board,
                             vector<string>& words) {
        m = board.size();
        n = board[0].size();
        ans.clear();

        // Build the Trie containing all words.
        for (const string& word : words) {
            TrieNode* curr = &root;

            for (char c : word) {
                int idx = c - 'a';

                if (!curr->children[idx]) {
                    curr->children[idx] = make_unique<TrieNode>();
                }

                curr = curr->children[idx].get();
            }

            curr->word = word;
        }

        // Start DFS from every cell on the board.
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                dfs(board, r, c, &root);
            }
        }

        return ans;
    }
};
```

---

## 6. Review & Takeaways

### Step-by-Step Dry Run

Board:

```text
o a a n
e t a e
i h k r
i f l v
```

`words = ["oath", "pea", "eat", "rain"]`, so the Trie is:

```text
root
 ├─ o -> a -> t -> h*      (* = complete word)
 ├─ p -> e -> a*
 ├─ e -> a -> t*
 └─ r -> a -> i -> n*
```

```text
1. Start at (0,0) = 'o':
   root has child 'o' -> node(o).
   '(o)' has no word.
   '#' mark (0,0), explore neighbours.

2. (1,0) = 'e':
   node(o) has child 'e' -> node(oe).
   ... continue o -> a -> t -> h:
   (0,1)='a', (0,2)='a'? branch follows only valid Trie prefix.
   Reach 'h' at (2,1)? path o(0,0) -> a(0,1) -> t(1,1) -> h(2,1):
   node(oath).word == "oath" -> emit "oath", clear word.

3. After unwinding, node(oath).isEmpty() == true
   -> parent->children['h'].reset() removes the dead branch.

4. Start at (1,1) = 't', (2,0) = 'i', (0,1) = 'a', ... 
   each start reuses the pruned Trie; "eat" found via e -> a -> t.

5. "pea": 'p' does not exist on the board at a valid path
   -> 'p' branch is pruned when its DFS attempts fail.

6. "rain" requires r -> a -> i -> n; board gives r(2,3) with no valid
   'a' continuation in order -> never emitted.

Result: ["oath", "eat"]
```

---

### The Architectural Pattern

```text
        All words -> build ONE Trie
                         ↓
        Each terminal node stores the FULL word
                         ↓
     For every board cell -> dfs(cell, root, path)
                         ↓
   ┌───────────────────────────────────────────────┐
   │ child missing?      -> prune (return)          │  prefix cut-off
   │ node.word non-empty -> emit + clear word       │  dedupe in O(1)
   │ mark '#' -> recurse 4 dirs -> restore          │  backtracking
   │ node.isEmpty()?     -> parent->child.reset()   │  branch pruning
   └───────────────────────────────────────────────┘
```

* **Next Review Date:**
* **Key Takeaway:** To find many words in a grid, do not DFS per word — build one Trie so every step is prefix-aware, store the whole word at the terminal node for $O(1)$ output and dedupe, and delete empty branches while backtracking so the Trie shrinks as the search progresses.
