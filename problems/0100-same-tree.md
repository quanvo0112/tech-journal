# 0100. Same Tree

- **Problem Link:** https://leetcode.com/problems/same-tree/
- **Difficulty:** `Easy`
- **NeetCode Category:** `Trees`
- **LeetCode Topics:** `Tree` / `Depth-First Search` / `Breadth-First Search` / `Binary Tree`
- **Core Pattern:** `Recursive DFS / Dual Tree Pairwise Structural & Value Matching`
- **Last Practiced:** 2026-10-04
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given the roots of two binary trees `p` and `q`, write a function to check if they are the same or not.
  - Two binary trees are considered the same if they are **structurally identical**, and the nodes have the **same value**.
  - Constraints: Number of nodes in both trees is in $[0, 100]$, node values in $[-10^4, 10^4]$.
- **Core Intuition:**
  - Two trees rooted at $p$ and $q$ are identical if and only if:
    1. Both root nodes are `nullptr` (empty subtrees match).
    2. Neither root is `nullptr`, their values are equal ($p.\text{val} == q.\text{val}$), their left subtrees are identical, **and** their right subtrees are identical:
       $$\text{isSameTree}(p, q) = (p.\text{val} == q.\text{val}) \land \text{isSameTree}(p.\text{left}, q.\text{left}) \land \text{isSameTree}(p.\text{right}, q.\text{right})$$
  - Any mismatch in structure (one is `nullptr` while the other is not) or in value immediately falsifies equality.
  - **Natural Short-Circuit Pruning:**
    - Using boolean short-circuit evaluation (`&&` in C++), if the left subtree fails matching, the recursion immediately halts and returns `false` without traversing the right subtree.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Iterative BFS / Level-Order with Queues ($O(N)$ time, $O(W)$ space):**
   - Push pairs `(p, q)` into a `std::queue<pair<TreeNode*, TreeNode*>>`.
   - Pop pairs, compare nullability and values, then push children `(p->left, q->left)` and `(p->right, q->right)`.
   - *Trade-off:* Avoids call stack depth, but requires explicit queue allocations and verbose boilerplates.

2. **Approach 2 — Pairwise Recursive DFS (Chosen Optimal Solution):**
   - Base Case 1 (Both null): `if (!p && !q) return true;`
   - Base Case 2 (Structural or Value Mismatch): `if (!p || !q || p->val != q->val) return false;`
   - Recursive Case: `return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);`
   - *Verdict:* Optimal $O(\min(N, M))$ time and $O(\min(h_p, h_q))$ auxiliary stack space. Extremely concise, zero extra memory allocations, and automatic short-circuiting.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(\min(N, M))$
  - Where $N$ and $M$ are the number of nodes in trees $p$ and $q$ respectively.
  - In the worst case (both trees are identical), every corresponding pair of nodes is visited exactly once, performing constant $O(1)$ comparisons.
  - If a mismatch occurs early, short-circuiting terminates traversal immediately.
- **Space Complexity:** $O(\min(h_p, h_q))$
  - Where $h_p$ and $h_q$ are the heights of the two binary trees, determining the maximum recursion stack depth.
  - **Balanced Trees:** $O(\log N)$ stack frames.
  - **Skewed / Degenerate Trees:** $O(N)$ stack frames.
  - Auxiliary heap memory is strictly $O(1)$.

---

## 4. Edge Cases & Gotchas

- [x] **Both Trees Empty (`!p && !q`):** Handled by `if (!p && !q) return true;` (two null pointers are structurally identical).
- [x] **One Tree Empty, One Non-Empty:** Caught by `!p || !q`, returning `false`.
- [x] **Same Values, Different Structure:** E.g., Tree 1 has left child 2, Tree 2 has right child 2. Pairwise traversal aligns `(p->left, q->left)` and detects that one is null while the other is not, correctly returning `false`.
- [x] **Order of Base Checks:** Checking `(!p && !q)` *before* accessing `p->val` or `q->val` prevents null-pointer dereferencing errors.

---

## 5. Clean Code (Optimal Solution: Pairwise Recursive DFS)

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
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        // Both nodes are null -> identical subtrees
        if (!p && !q) {
            return true;
        }

        // Structural mismatch (one null, one not) or value mismatch
        if (!p || !q || p->val != q->val) {
            return false;
        }

        // Both nodes match; recursively verify both child branches
        return isSameTree(p->left, q->left) &&
               isSameTree(p->right, q->right);
    }
};
```

---

## 6. Review & Takeaways

### Visual Dual-Tree Matching Dry-Run

#### Case 1: Identical Trees

```text
p:        1                 q:        1
         / \                         / \
        2   3                       2   3

1. Compare root p(1) and q(1):
   - Both non-null, 1 == 1 -> Proceed to children
2. Recurse Left: Compare p(2) and q(2):
   - Both non-null, 2 == 2 -> Children are both null -> true
3. Recurse Right: Compare p(3) and q(3):
   - Both non-null, 3 == 3 -> Children are both null -> true
4. Combine: true && true -> Return true
```

#### Case 2: Structural Mismatch (Different Topologies)

```text
p:        1                 q:        1
         /                             \
        2                               2

1. Compare root p(1) and q(1): values equal (1 == 1)
2. Recurse Left: Compare p->left (2) and q->left (null):
   - p exists, q is null -> !p || !q triggers -> Return false
3. Short-circuit: right branch is never visited -> Return false
```

---

### The Architectural Mental Model

```text
                           Same Tree: isSameTree(p, q)
                                        ↓
                                 !p && !q ?
                                   /    \
                             YES  /      \  NO
                                 ↓        ↓
                            return true   !p || !q || p->val != q->val ?
                                              /          \
                                        YES  /            \  NO
                                            ↓              ↓
                                       return false   Recurse Both Branches:
                                                      isSameTree(p->left, q->left)
                                                                 &&
                                                      isSameTree(p->right, q->right)
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** Dual-tree comparison decomposes recursively into: same current node value + identical left subtrees + identical right subtrees, leveraging boolean short-circuiting for free early termination.
