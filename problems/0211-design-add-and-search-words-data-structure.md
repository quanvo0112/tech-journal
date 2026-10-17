# 0211. Design Add and Search Words Data Structure

- **Problem Link:** https://leetcode.com/problems/design-add-and-search-words-data-structure/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Tries`
- **LeetCode Topics:** `String` / `Depth-First Search` / `Design` / `Trie`
- **Core Pattern:** `Trie + DFS Backtracking with Wildcard '.'`
- **Last Practiced:**
- **Proficiency Level:**
  - [ ] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - *"Design a data structure that supports adding new words and finding if a string matches any previously added string."*
  - *"`search(word)` can contain dots `'.'` where each dot can match any single letter."*
  - *"All strings consist of lowercase English letters (and `'.'` in search queries)."*
- **Why naive storage falls short:**
  - Storing words in an `unordered_set<string>` answers exact `search(word)` in $O(L)$, but a pattern like `".ad"` has no single key to look up — you would have to compare against every stored word, giving $O(N \cdot L)$ per wildcard query.
  - Iterative prefix walking (the technique from `0208`) also fails, because a `'.'` forces a **branch** into every existing child rather than following a single edge.
- **Core intuition:**
  - Reuse the `0208` Trie to share common prefixes as tree paths (each node = one prefix, each edge = one next character).
  - For a literal character, follow exactly one edge — no ambiguity.
  - For `'.'`, the search becomes a **DFS with branching**: try every existing child, and accept if *any* branch reaches a complete word.
  - Fundamental rule:
    $$\text{search}("p") \text{ succeeds} \iff \exists \text{ a path matching } p \text{ ending at a node with } isEnd = true.$$
  - The terminal check `isEnd` is mandatory: reaching the end of the query only proves the prefix exists, not that a full word was stored.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Naive Hash Set of Words ($O(L)$ Add, $O(N \cdot L)$ Search):**
   - Insert each word into an `unordered_set<string>`.
   - For a query with no `'.'`, a direct set lookup is $O(L)$.
   - For a query with `'.'`, iterate all $N$ stored words and match character-by-character.
   - *Verdict:* Unacceptable once wildcards appear; completely ignores shared-prefix structure.

2. **Approach 2 — Hash-Map Children Trie (`unordered_map<char, Node*>` per Node):**
   - Each node stores only the outgoing edges it uses.
   - *Trade-off:* Memory-efficient for sparse alphabets, but pays hashing overhead and pointer chasing on every step. Overkill for a fixed 26-letter lowercase alphabet.

3. **Approach 3 — Fixed 26-Array Children + `unique_ptr` + `isEnd` + DFS `searchFrom` (Optimal & Implemented):**
   - Each node holds `array<unique_ptr<Node>, 26> children` plus `bool isEnd = false`.
   - `addWord(word)`: identical to `0208` `insert` — walk/create one child per character, then set `isEnd = true` on the terminal node.
   - `search(word)`: delegate to a recursive helper `searchFrom(node, word, index)`:
     - Base case: `index == word.size()` -> return `node->isEnd`.
     - If `word[index] == '.'`: loop over all 26 child slots and recurse into each non-null child; return `true` on the first success, otherwise `false`.
     - Otherwise: descend only into `children[c - 'a']`; if it is null return `false`, else recurse.
   - *Why this variant:* literal characters stay $O(1)$ via direct indexing `c - 'a'`, while `'.'` is handled by a compact, self-contained DFS. `unique_ptr` gives automatic subtree cleanup with no manual `delete`. This is the direct bridge from `0208` to `0212` (Word Search II).

---

## 3. Complexity Analysis

- **Time Complexity:**
  - `addWord()`: $O(L)$ — Explanation: exactly one array-index step and (at most) one node creation per character.
  - `search()`: $O(\Sigma^d \cdot L)$ in the worst case, where $d$ is the number of `'.'` in the query and $\Sigma = 26$ — Explanation: each `'.'` can branch into up to 26 children; with the problem constraint $d \le 2$ this bounds to $O(26^2 \cdot L)$. Without wildcards it degrades to the $O(L)$ walk of `0208`.
- **Space Complexity:** $O(S \cdot \Sigma) \approx O(S)$ nodes — Explanation: $S$ is the total number of characters across all inserted words (shared prefixes stored once); each node reserves 26 slots but only nodes along inserted paths are materialized. The recursive `search()` additionally uses $O(L)$ stack space (one frame per query character).

---

## 4. Edge Cases & Gotchas

- [ ] **Pattern shorter than a stored word (`addWord("bad")`, `search("ba")`):**
  - Path `b -> a` exists but `a.isEnd == false`, so `search("ba")` must return `false` — do not return `true` merely because the walk completed.
- [ ] **Wildcard matching a dead end (`addWord("bad")`, `search("p..")`):**
  - No child `p` exists at the root, so the DFS must return `false` after exhausting all branches.
- [ ] **Multiple wildcards (`search("b..")`, `search("...")`):**
  - Branching multiplies; ensure the base-case `isEnd` check only triggers after the *entire* query is consumed.
- [ ] **All-wildcard query that should match a word shorter than the query (`search("....")` on `"bad"`):**
  - Length mismatch is naturally rejected by the base case, since every wildcard must consume a real character.
- Common pitfalls during implementation:
  - Returning `true` immediately when the query is fully consumed without checking `isEnd` (turns `search` into `startsWith`).
  - Forgetting to skip null child slots while iterating with a range-`for` over `unique_ptr` (must check `if (child)` before calling `child.get()`).
  - Mixing recursion and manual memory: never `delete` inside the DFS — `unique_ptr` owns the subtree.

---

## 5. Clean Code

```cpp
#include <array>
#include <memory>
#include <string>

using namespace std;

class WordDictionary {
private:
    struct Node {
        array<unique_ptr<Node>, 26> children{};
        bool isEnd = false;
    };

    Node root;

    bool searchFrom(Node* node, const string& word, int index) {
        if (index == word.size()) {
            return node->isEnd;
        }

        char c = word[index];

        if (c == '.') {
            // Try every existing child.
            for (auto& child : node->children) {
                if (child &&
                    searchFrom(child.get(), word, index + 1)) {
                    return true;
                }
            }

            return false;
        }

        auto& child = node->children[c - 'a'];

        if (!child) {
            return false;
        }

        return searchFrom(child.get(), word, index + 1);
    }

public:
    WordDictionary() = default;

    void addWord(string word) {
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
        return searchFrom(&root, word, 0);
    }
};
```

---

## 6. Review & Takeaways

### Step-by-Step Dry Run

Operations: `addWord("bad") -> addWord("dad") -> addWord("mad") -> search("pad") -> search("bad") -> search(".ad") -> search("b..") -> search("ba")`

```text
Trie after adds:
root
 ├─ b -> a -> d*        (* = isEnd)
 ├─ d -> a -> d*
 └─ m -> a -> d*

1. search("pad"):
   root has no child 'p' -> return false

2. search("bad"):
   follow b -> a -> d, d.isEnd == true -> return true

3. search(".ad"):
   '.' branches into b, d, m
     branch b -> a -> d, isEnd == true -> return true

4. search("b.."):
   follow b -> then '.' -> '.' -> reaches d at index 3, isEnd == true -> return true

5. search("ba"):
   follow b -> a, index == 2 == word.size(), a.isEnd == false -> return false
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
   addWord(word):
     walk/create child per char -> set isEnd = true
                     ↓
   search(word) -> searchFrom(node, word, i):
     ├─ i == len        -> return node->isEnd          (terminal check is mandatory)
     ├─ word[i] == '.'  -> DFS into every existing child (branching)
     └─ literal c       -> descend child[c - 'a'] only  (single edge)
```

* **Next Review Date:**
* **Key Takeaway:** Literal characters walk a single Trie edge, but `'.'` forces a DFS that branches into every existing child — and the search is only successful when a fully consumed query lands on a node with `isEnd == true`.
