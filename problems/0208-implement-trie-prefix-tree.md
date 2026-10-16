# 0208. Implement Trie (Prefix Tree)

- **Problem Link:** https://leetcode.com/problems/implement-trie-prefix-tree/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Tries`
- **LeetCode Topics:** `Hash Table` / `String` / `Design` / `Trie`
- **Core Pattern:** `Trie (Prefix Tree) with Fixed 26-Array Children + isEnd Terminal Flag`
- **Last Practiced:** 2026-10-16
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - *"Implement a Trie with `insert`, `search`, and `startsWith` methods."*
  - *"All strings consist of lowercase English letters `a-z`."*
  - Queries ask both *"is this an exact stored word?"* and *"is this a prefix of any stored word?"*.
- **Why naive storage falls short:**
  - Storing full words in a hash set answers `search(word)` in $O(L)$ but answers `startsWith(prefix)` only by scanning every stored word — $O(N \cdot L)$ per query, where $N$ is the number of words.
  - A sorted list plus binary search still cannot share work across words with common prefixes (e.g. `"app"`, `"apple"`, `"apply"` re-scan `"app"` every time).
- **Core intuition:**
  - Share common prefixes as shared tree paths. Each node represents one prefix; each edge represents one next character.
  - Fundamental rule:
    $$"A word exists \iff its path exists AND the terminal node is marked isEnd."$$
  - A prefix exists iff its path exists — no terminal check needed. That single-flag difference is the entire distinction between `search()` and `startsWith()`.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Naive Hash Set of Words ($O(L)$ Insert/Search, $O(N \cdot L)$ Prefix Check):**
   - Insert each word into an `unordered_set<string>`.
   - `search(word)` is a direct set lookup.
   - `startsWith(prefix)` must iterate all $N$ words and test `word.starts_with(prefix)`.
   - *Verdict:* Unacceptable once prefix queries are frequent; ignores all shared-prefix structure.

2. **Approach 2 — Hash-Map Children Trie (`unordered_map<char, Node*>` per Node):**
   - Each node stores only the outgoing edges it actually uses.
   - *Trade-off:* Memory-efficient for sparse alphabets (e.g. Unicode), but every step pays hashing overhead and pointer chasing. Overkill for a fixed 26-letter lowercase alphabet.

3. **Approach 3 — Fixed 26-Array Children + `unique_ptr` + `isEnd` Flag (Optimal & Implemented):**
   - Each node holds `array<unique_ptr<Node>, 26> children` plus `bool isEnd = false`.
   - `insert(word)`: walk/create one child per character, then set `isEnd = true` on the terminal node.
   - `search(word)`: walk the path; return `false` on any missing child, otherwise return `curr->isEnd`.
   - `startsWith(prefix)`: walk the path; return `false` on any missing child, otherwise return `true` (no `isEnd` check).
   - *Why this variant:* direct index `c - 'a'` gives $O(1)$ branching with zero hashing; `unique_ptr` gives automatic memory cleanup of the whole subtree with no manual `delete`. This is the canonical foundation for `0211` (wildcard search) and `0212` (board DFS + Trie pruning).

---

## 3. Complexity Analysis

- **Time Complexity:** $O(L)$ per operation, where $L$ is the length of `word` / `prefix` — Explanation: each method performs exactly one array-index step per character, no backtracking.
- **Space Complexity:** $O(S \cdot \Sigma)$ in the worst case, effectively $O(S)$ nodes — Explanation: $S$ is the total number of characters across all inserted words (shared prefixes stored once); each node reserves 26 child slots but only materializes nodes along inserted paths. Each `search` / `startsWith` uses $O(1)$ auxiliary space (iterative pointer walk).

---

## 4. Edge Cases & Gotchas

- [x] **Prefix vs. whole word (`insert("apple")`, then `search("app")` vs `startsWith("app")`):**
  - Path `a -> p -> p` exists but node `p` has `isEnd == false`, so `search("app")` returns `false` while `startsWith("app")` returns `true`.
- [x] **Inserting a word that is a prefix of an existing word (`insert("app")` after `insert("apple")`):**
  - Only flips `isEnd = true` on the existing `p` node; must not create duplicate nodes or disturb the longer word.
- [x] **Inserting a duplicate word:**
  - Re-walking the same path and re-setting `isEnd = true` is idempotent and safe.
- [x] **Empty-string input:**
  - Constraints guarantee `1 <= word.length, prefix.length <= 2000`, so no empty input handling is required; the root node itself is never marked as a word.
- Common pitfalls during implementation:
  - Forgetting the `isEnd` check in `search()` (degenerates into `startsWith()`).
  - Using raw `new` without `delete` — `unique_ptr` avoids the leak entirely.
  - Indexing with `c - 'a'` without assuming lowercase input — safe here per constraints.

---

## 5. Clean Code

```cpp
#include <array>
#include <memory>
#include <string>

using namespace std;

class Trie {
private:
    struct Node {
        array<unique_ptr<Node>, 26> children{};
        bool isEnd = false;
    };

    Node root;

public:
    Trie() = default;

    void insert(string word) {
        Node* curr = &root;

        for (char c : word) {
            int idx = c - 'a';

            if (!curr->children[idx]) {
                curr->children[idx] = make_unique<Node>();
            }

            curr = curr->children[idx].get();
        }

        curr->isEnd = true;
    }

    bool search(string word) {
        Node* curr = &root;

        for (char c : word) {
            int idx = c - 'a';

            if (!curr->children[idx]) {
                return false;
            }

            curr = curr->children[idx].get();
        }

        return curr->isEnd;
    }

    bool startsWith(string prefix) {
        Node* curr = &root;

        for (char c : prefix) {
            int idx = c - 'a';

            if (!curr->children[idx]) {
                return false;
            }

            curr = curr->children[idx].get();
        }

        return true;
    }
};
```

---

## 6. Review & Takeaways

### Step-by-Step Dry Run

Operations: `insert("apple") -> search("apple") -> search("app") -> startsWith("app") -> insert("app") -> search("app")`

```text
1. insert("apple"):
   root -> a -> p -> p -> l -> e
                                    (e.isEnd = true)

2. search("apple"):
   walk a -> p -> p -> l -> e, all edges exist, e.isEnd == true
   return true

3. search("app"):
   walk a -> p -> p, all edges exist, but p.isEnd == false
   return false

4. startsWith("app"):
   walk a -> p -> p, all edges exist, no isEnd check
   return true

5. insert("app"):
   walk existing a -> p -> p, set p.isEnd = true
   root -> a -> p* -> p -> l -> e
              (* = isEnd now true)

6. search("app"):
   walk a -> p -> p, p.isEnd == true
   return true
```

---

### The Architectural Pattern

```text
              Trie Node
                  ↓
      ┌───────────────────────┐
      │ children[26] │ isEnd  │
      └───────────────────────┘
                  ↓
   insert(word):
     walk/create child per char -> set isEnd = true
                  ↓
   search(word):
     walk path -> return isEnd
                  ↓
   startsWith(prefix):
     walk path -> return true
```

* **Next Review Date:**
* **Key Takeaway:** A Trie shares prefixes as tree paths and separates "path exists" (`startsWith`) from "path exists AND terminates here" (`search` via `isEnd`) — that one flag is the whole problem.
