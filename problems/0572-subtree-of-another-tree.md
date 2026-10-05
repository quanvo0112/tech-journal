# 0572. Subtree of Another Tree

- **Problem Link:** https://leetcode.com/problems/subtree-of-another-tree/
- **Difficulty:** `Easy`
- **NeetCode Category:** `Trees`
- **LeetCode Topics:** `Tree` / `Depth-First Search` / `String Matching` / `Binary Tree` / `Hash Function`
- **Core Pattern:** `Recursive DFS / Subtree Equivalence Verification (isSameTree Primitive)`
- **Last Practiced:** 2026-10-05
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given the roots of two binary trees `root` and `subRoot`, return `true` if there is a subtree of `root` with the same structure and node values of `subRoot`, and `false` otherwise.
  - A **subtree** of a binary tree `tree` is a tree that consists of a node in `tree` and **all of this node's descendants**. The tree `tree` could also be considered as a subtree of itself.
  - Constraints: Number of nodes in `root` is in $[1, 2000]$, `subRoot` in $[1, 1000]$. Node values in $[-10^4, 10^4]$.
- **The Core Invariant:**
  - A tree $T_{\text{sub}}$ is a subtree of $T$ if and only if there exists at least one node $u \in T$ such that the rooted subtree at $u$ is **strictly identical** in both topology and values to $T_{\text{sub}}$.
  - Having identical node values (`root->val == subRoot->val`) is merely a preliminary candidate condition, not sufficient on its own. The entire subtree topology and all descendant leaf nodes must match.
- **Pattern Synthesis (Direct Reuse of LeetCode 0100 Same Tree):**
  - This problem decomposes cleanly into two cooperating recursive DFS functions:
    1. **Outer Traversal (`isSubtree`):** Iterates over every node in `root` as a potential subtree root candidate.
    2. **Inner Verification (`isSameTree`):** Validates structural and numerical equivalence between the candidate subtree and `subRoot`.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Tree Serialization & String Matching / KMP ($O(M + N)$ time, $O(M + N)$ space):**
   - Serialize both trees into unique preorder traversal strings with dedicated delimiter symbols and null markers (e.g., `,#` for null).
   - Run the Knuth-Morris-Pratt (KMP) or Rabin-Karp substring search algorithm to verify if `serialize(subRoot)` is a substring of `serialize(root)`.
   - *Trade-off:* Achieves linear time $O(M + N)$, but requires heavy string allocations, delimiter parsing, and significant boilerplate overhead.

2. **Approach 2 — Recursive DFS with `isSameTree` (Chosen Optimal Solution):**
   - **Base Cases:**
     - `if (!subRoot) return true;` (an empty tree is a valid subtree of any tree).
     - `if (!root) return false;` (`root` is exhausted while `subRoot` is non-empty).
   - **Candidate Matching:**
     - If `isSameTree(root, subRoot)` evaluates to `true`, the subtree is found.
     - Otherwise, recursively search both children:
       $$\text{isSubtree}(\text{root}, \text{subRoot}) = \text{isSubtree}(\text{root.left}, \text{subRoot}) \lor \text{isSubtree}(\text{root.right}, \text{subRoot})$$
   - *Verdict:* Clean, declarative, modular, and directly reuses the `isSameTree` primitive with zero dynamic memory allocation overhead.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(M \times N)$ worst-case
  - Where $M$ is the number of nodes in `root` and $N$ is the number of nodes in `subRoot`.
  - In the worst case (e.g., both trees are large, identical repeating paths of identical values that only differ at the very last leaf), `isSameTree` traverses up to $O(N)$ nodes for each of the $M$ candidate nodes in `root`.
  - In average cases, mismatches at the root or early nodes terminate in $O(1)$ via boolean short-circuiting.
- **Space Complexity:** $O(H)$
  - Where $H$ is the height of the main tree `root`, corresponding to the maximum depth of the call stack.
  - **Balanced Tree:** $H = O(\log M)$.
  - **Degenerate / Skewed Tree:** $H = O(M)$.
  - Auxiliary heap memory is strictly $O(1)$.

---

## 4. Edge Cases & Gotchas

- [x] **Empty `subRoot` (`subRoot == nullptr`):** Handled by `if (!subRoot) return true;`. By definition, an empty tree is a trivial subtree of any tree.
- [x] **Empty `root` (`root == nullptr` with non-empty `subRoot`):** Handled by `if (!root) return false;`. A non-empty tree cannot be contained within an empty tree.
- [x] **Identical Values with Different Topologies:** E.g., `root` has node with value `4` and right child `2`, while `subRoot` has `4` with left child `2`. `isSameTree` checks left-with-left and right-with-right, correctly identifying the structural mismatch and returning `false`.
- [x] **Partial Matching vs. Full Subtree:** A candidate node in `root` may match all nodes of `subRoot` but have additional children below its leaves. Since a subtree must include **all** descendants, `isSameTree` checks that child nodes are null at identical locations, properly rejecting partial matches.

---

## 5. Clean Code (Optimal Solution: Recursive DFS + `isSameTree`)

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
private:
    // Helper: Validates structural and numerical equivalence (LeetCode 100)
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if (!p && !q) {
            return true;
        }

        if (!p || !q || p->val != q->val) {
            return false;
        }

        return isSameTree(p->left, q->left) &&
               isSameTree(p->right, q->right);
    }

public:
    bool isSubtree(TreeNode* root, TreeNode* subRoot) {
        // An empty tree is a subtree of every tree
        if (!subRoot) {
            return true;
        }

        // Exhausted main tree without finding a matching subtree
        if (!root) {
            return false;
        }

        // Check if current node is the root of an identical subtree
        if (isSameTree(root, subRoot)) {
            return true;
        }

        // Otherwise, search recursively in left or right branch
        return isSubtree(root->left, subRoot) ||
               isSubtree(root->right, subRoot);
    }
};
```

---

## 6. Review & Takeaways

### Visual Dry-Run Tracing

```text
Given Trees:

root:                 subRoot:
        3                   4
       / \                 / \
      4   5               1   2
     / \
    1   2

Step-by-Step Traversal:
1. Examine Node 3:
   - isSameTree(3, 4) -> 3 != 4 -> false
2. Recurse Left to Node 4:
   - isSameTree(4, 4):
     - Roots match: 4 == 4
     - Compare left subtrees:  isSameTree(1, 1) -> 1 == 1, children null -> true
     - Compare right subtrees: isSameTree(2, 2) -> 2 == 2, children null -> true
     - Combined: true && true -> true
3. Early Return:
   - isSameTree(4, subRoot) returns true
   - Short-circuit || avoids exploring right subtree (Node 5)

Final Result: true
```

---

### The Architectural Mental Model

```text
                     Subtree of Another Tree: isSubtree(root, subRoot)
                                            ↓
                                     !subRoot ?
                                       /   \
                                 YES  /     \  NO
                                     ↓       ↓
                                 return true  !root ?
                                               /   \
                                         YES  /     \  NO
                                             ↓       ↓
                                        return false  isSameTree(root, subRoot) ?
                                                          /          \
                                                    YES  /            \  NO
                                                        ↓              ↓
                                                   return true    Branch Recursively:
                                                                  isSubtree(root->left, subRoot)
                                                                               ||
                                                                  isSubtree(root->right, subRoot)
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** Treat every node in the parent tree as a potential subtree origin; invoke the `isSameTree` structural validator at each candidate and short-circuit via boolean `||`.
