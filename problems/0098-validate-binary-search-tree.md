# 0098. Validate Binary Search Tree

- **Problem Link:** https://leetcode.com/problems/validate-binary-search-tree/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Trees`
- **LeetCode Topics:** `Tree` / `Depth-First Search` / `Binary Search Tree` / `Binary Tree`
- **Core Pattern:** `Top-Down DFS / Interval Constraint Propagation (Lower & Upper Bounds)`
- **Last Practiced:** 2026-10-06
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given the `root` of a binary tree, determine if it is a valid binary search tree (BST).
  - A valid BST is defined as follows:
    1. The left subtree of a node contains only nodes with keys **strictly less than** the node's key.
    2. The right subtree of a node contains only nodes with keys **strictly greater than** the node's key.
    3. Both the left and right subtrees must also be binary search trees.
  - Constraints: The number of nodes is in $[1, 10^4]$, and node values span the entire 32-bit signed integer range $[-2^{31}, 2^{31} - 1]$ (`[INT_MIN, INT_MAX]`).
- **The Classic Local-Check Fallacy:**
  - A common bug is checking only immediate child relationships:
    $$\text{node}->\text{left}->\text{val} < \text{node}->\text{val} < \text{node}->\text{right}->\text{val}$$
  - Consider the tree:
    ```text
          5
         / \
        1   7
           / \
          4   8
    ```
    Here, $4 < 7$ and $8 > 7$, so every parent-child pair appears valid locally. However, $4$ is in the right subtree of $5$, violating the global BST requirement that all nodes in the right subtree must exceed $5$ ($4 \ngtr 5$).
- **The Ancestral Interval Invariant:**
  - Every node in a BST inherits a valid open interval $(\text{lower}, \text{upper})$ formed cumulatively by **all ancestors** along the path from the root.
  - At `root`, the valid range is unrestricted: $(-\infty, +\infty)$.
  - When branching **left**, all descendants must be strictly smaller than the current node, tightening the upper bound:
    $$(\text{lower}, \text{node}->\text{val})$$
  - When branching **right**, all descendants must be strictly greater than the current node, tightening the lower bound:
    $$(\text{node}->\text{val}, \text{upper})$$

---

## 2. Approach & Trade-offs

1. **Approach 1 — Inorder Traversal with Previous Pointer ($O(N)$ time, $O(H)$ space):**
   - Traverse the tree in inorder (`Left -> Root -> Right`). In a valid BST, this yields a strictly increasing sorted sequence.
   - Maintain a pointer or variable tracking `prev`. If `node->val <= prev`, return `false`.
   - *Trade-off:* Valid and standard, but relies on traversing in an indirect projection. Iterative stacks or tracking previous state across stack unwinds introduces subtle pointer state management.

2. **Approach 2 — Recursive Top-Down DFS with `long long` Bounds (Chosen Optimal Solution):**
   - Traverse the tree recursively, passing the valid interval $(\text{lower}, \text{upper})$ downward.
   - If `!node`, return `true` (an empty tree is trivially a valid BST).
   - If `node->val <= lower || node->val >= upper`, return `false`.
   - Recurse into subtrees:
     $$\text{validate}(\text{left}, \text{lower}, \text{node}->\text{val}) \land \text{validate}(\text{right}, \text{node}->\text{val}, \text{upper})$$
   - *Verdict:* Optimal $O(N)$ time and $O(H)$ auxiliary space. Directly enforces the mathematical definition of a BST, isolates constraints cleanly, and eliminates extra memory overhead.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(N)$
  - Where $N$ is the number of nodes in the binary tree.
  - Each node is visited at most once. At each step, bound comparisons take $O(1)$ time. Early termination halts traversal the moment an invalid node is encountered.
- **Space Complexity:** $O(H)$
  - Where $H$ is the height of the tree.
  - Space is dictated solely by the recursive call stack frames.
  - Balanced tree: $O(\log N)$ stack depth.
  - Skewed tree: $O(N)$ stack depth in the worst case.

---

## 4. Edge Cases & Gotchas

- [x] **32-Bit Integer Boundary Values (`INT_MIN`, `INT_MAX`):** Node values can legally equal $-2^{31}$ or $2^{31}-1$. If bounds were initialized with 32-bit integers (`int lower = INT_MIN`), checking `node->val <= lower` would incorrectly flag a legitimate node whose value is `INT_MIN` as invalid. Using `long long` with `LLONG_MIN` and `LLONG_MAX` safely places the initial boundaries outside the valid 32-bit range.
- [x] **Strict Inequality (No Duplicates Allowed):** A standard LeetCode BST requires **strictly** less than and **strictly** greater than. Duplicate keys (e.g. $[2, 2, 2]$) violate the BST invariant. The check `node->val <= lower || node->val >= upper` correctly rejects duplicate values.
- [x] **Single-Node Tree:** Returns `true` since $\text{LLONG\_MIN} < \text{root}->\text{val} < \text{LLONG\_MAX}$.

---

## 5. Clean Code

```cpp
#include <climits>

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
    bool validate(TreeNode* node, long long lower, long long upper) {
        if (!node) {
            return true;
        }

        // The node value must fall strictly within (lower, upper).
        if (node->val <= lower || node->val >= upper) {
            return false;
        }

        // Left branch tightens upper bound; right branch tightens lower bound.
        return validate(node->left, lower, node->val) &&
               validate(node->right, node->val, upper);
    }

public:
    bool isValidBST(TreeNode* root) {
        return validate(root, LLONG_MIN, LLONG_MAX);
    }
};
```

---

## 6. Visual Walkthrough & Architectural Pattern

### Step-by-Step Execution Trace

#### Counterexample: Invalid BST $[5, 1, 7, \text{null}, \text{null}, 4, 8]$
```text
          5
         / \
        1   7
           / \
          4   8
```

1. **Root 5:** Valid range $(-\infty, +\infty)$.
   - $5 \in (-\infty, +\infty)$ $\rightarrow$ Valid.
   - Recurse left: `validate(1, -∞, 5)`.
   - Recurse right: `validate(7, 5, +∞)`.

2. **Left Child 1:** Valid range $(-\infty, 5)$.
   - $1 \in (-\infty, 5)$ $\rightarrow$ Valid.
   - Children are null $\rightarrow$ Returns `true`.

3. **Right Child 7:** Valid range $(5, +\infty)$.
   - $7 \in (5, +\infty)$ $\rightarrow$ Valid.
   - Recurse left: `validate(4, 5, 7)`.
   - Recurse right: `validate(8, 7, +∞)`.

4. **Node 4:** Valid range $(5, 7)$.
   - $4 \le 5$ (`node->val <= lower`) $\rightarrow$ **Violation detected!**
   - Returns `false` immediately without exploring further.

---

### The Architectural Mental Model

```text
                     Top-Down Range Constraint Propagation
                                       ↓
                        validate(node, lower, upper)
                                       ↓
                                 node == null? ─── Yes ───> return true
                                       ↓ No
                       node->val <= lower || node->val >= upper?
                                       │
                       ┌───────────────┴───────────────┐
                       │ Yes                           │ No
                       ↓                               ↓
                  return false         validate(left, lower, node->val)
                                                       &&
                                       validate(right, node->val, upper)
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** A valid BST requires every node to obey the cumulative boundaries imposed by all its ancestors; propagating $(\text{lower}, \text{upper})$ downward directly enforces this global invariant in $O(N)$ time.
