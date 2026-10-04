# 0110. Balanced Binary Tree

- **Problem Link:** https://leetcode.com/problems/balanced-binary-tree/
- **Difficulty:** `Easy`
- **NeetCode Category:** `Trees`
- **LeetCode Topics:** `Tree` / `Depth-First Search` / `Binary Tree`
- **Core Pattern:** `Post-Order Recursive DFS / Bottom-Up Height Validation with -1 Sentinel`
- **Last Practiced:** 2026-10-04
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given a binary tree, determine if it is **height-balanced**.
  - A **height-balanced** binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differs by more than one:
    $$|\text{height}(\text{left}) - \text{height}(\text{right})| \le 1 \quad \forall \text{ node} \in T$$
  - Constraints: The number of nodes in the tree is in $[0, 5000]$, node values in $[-10^4, 10^4]$.
- **Why Naive Top-Down Approaches Fall Short:**
  - In a naive top-down recursion, for every node we compute the full height of both its left and right subtrees (`height(node->left)`, `height(node->right)`), check if the difference is $\le 1$, and then recursively call `isBalanced` on both children.
  - On degenerate / skewed trees (e.g., a linked list shaped tree), calculating height repeatedly at every level takes $\sum_{i=1}^N i = O(N^2)$ time.
- **Core Intuition (Bottom-Up Post-Order Traversal):**
  - Instead of computing heights from top to bottom, compute heights from the leaves upward (**Bottom-Up**):
    $$\text{Leaves} \longrightarrow \text{Subtrees} \longrightarrow \text{Ancestors} \longrightarrow \text{Root}$$
  - A recursive call returns two pieces of information:
    1. A non-negative integer ($\ge 0$): The valid height of the subtree rooted at `node`.
    2. A sentinel value ($-1$): Indicates that an imbalance has already been detected somewhere within that subtree.
  - **Early Exit / Pruning:** As soon as any child returns $-1$, the entire subtree is invalid. The caller short-circuits immediately by returning $-1$ without needing to explore or compute heights for the remaining sibling subtrees.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Naive Top-Down DFS ($O(N^2)$ time, $O(h)$ space):**
   - For every node, compute `height(node->left)` and `height(node->right)`.
   - Check `abs(leftHeight - rightHeight) <= 1 && isBalanced(node->left) && isBalanced(node->right)`.
   - *Drawback:* Severe redundancy. The height of lower nodes is recalculated once for every ancestor above them, leading to quadratic time on unbalanced trees.

2. **Approach 2 — Bottom-Up DFS Returning Pair/Struct ($O(N)$ time, $O(h)$ space):**
   - Each recursive call returns `pair<bool, int>` representing `{is_balanced, height}`.
   - *Drawback:* Struct instantiation / unpacking overhead across every stack frame, even though a single integer can represent both states via a sentinel value.

3. **Approach 3 — Bottom-Up Post-Order DFS with `-1` Sentinel (Chosen Optimal Solution):**
   - Implement helper `checkHeight(node)`:
     - Base case: `if (!node) return 0;`
     - Recurse left: if `leftHeight == -1`, immediately `return -1;` (short-circuit).
     - Recurse right: if `rightHeight == -1`, immediately `return -1;` (short-circuit).
     - If `abs(leftHeight - rightHeight) > 1`, return `-1` (current node is unbalanced).
     - Otherwise, return valid subtree height: `1 + max(leftHeight, rightHeight)`.
   - The public method simply checks: `return checkHeight(root) != -1;`.
   - *Verdict:* Optimal $O(N)$ time and $O(h)$ auxiliary stack memory. Single traversal, zero recomputation, clean early termination.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(N)$
  - Where $N$ is the total number of nodes in the binary tree.
  - In the worst case (a perfectly balanced tree), every node is visited exactly once, performing $O(1)$ operations (`abs`, `max`, additions, comparisons).
  - On unbalanced trees, early pruning returns $-1$ as soon as an imbalance is found, visiting strictly fewer than $N$ nodes.
- **Space Complexity:** $O(h)$
  - Where $h$ is the height of the tree, determining the maximum depth of the call stack.
  - **Balanced Binary Tree:** $h = O(\log N)$ stack frames.
  - **Degenerate / Skewed Tree:** $h = O(N)$ stack frames.
  - Auxiliary heap allocation is strictly $O(1)$.

---

## 4. Edge Cases & Gotchas

- [x] **Empty Tree (`root == nullptr`):** Handled cleanly by the base condition `if (!node) return 0;`. Result evaluates to `0 != -1` which returns `true` (an empty tree is height-balanced by definition).
- [x] **Single-Node Tree:** Both child calls return `0`. `abs(0 - 0) = 0 <= 1`. Returns height `1 != -1`, returning `true`.
- [x] **Subtree Imbalance Far Down the Tree:** If a leaf's parent is unbalanced, returning `-1` bubbles up to the root without doing unnecessary work on other unrelated branches.
- [x] **Sentinel Collision Protection:** Since valid subtree heights are strictly non-negative ($\ge 0$), using `-1` as an error indicator guarantees no collision with valid tree heights.

---

## 5. Clean Code (Optimal Solution: Bottom-Up Post-Order DFS)

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
    int checkHeight(TreeNode* node) {
        if (!node) {
            return 0;
        }

        int leftHeight = checkHeight(node->left);
        if (leftHeight == -1) {
            return -1; // Early exit: left subtree is already unbalanced
        }

        int rightHeight = checkHeight(node->right);
        if (rightHeight == -1) {
            return -1; // Early exit: right subtree is already unbalanced
        }

        if (abs(leftHeight - rightHeight) > 1) {
            return -1; // Current node violates height-balanced invariant
        }

        return 1 + max(leftHeight, rightHeight);
    }

public:
    bool isBalanced(TreeNode* root) {
        return checkHeight(root) != -1;
    }
};
```

---

## 6. Review & Takeaways

### Visual Dry-Run Tracing

#### Example 1: Balanced Tree `[3, 9, 20, null, null, 15, 7]`

```text
        3
       / \
      9  20
         / \
        15  7

Bottom-Up Height Propagation:
- Leaf 9:   left=0, right=0  --> abs(0-0)=0 <= 1 --> return height = 1
- Leaf 15:  left=0, right=0  --> abs(0-0)=0 <= 1 --> return height = 1
- Leaf 7:   left=0, right=0  --> abs(0-0)=0 <= 1 --> return height = 1
- Node 20:  left=1, right=1  --> abs(1-1)=0 <= 1 --> return height = 1 + max(1,1) = 2
- Root 3:   left=1, right=2  --> abs(1-2)=1 <= 1 --> return height = 1 + max(1,2) = 3

Result: checkHeight(root) = 3 != -1  --> true (Balanced)
```

#### Example 2: Unbalanced Skewed Tree `[1, 2, 2, 3, 3, null, null, 4, 4]`

```text
        1
       /
      2
     /
    3

Bottom-Up Height Propagation:
- Leaf 3:   left=0, right=0  --> abs(0-0)=0 <= 1 --> return height = 1
- Node 2:   left=1, right=0  --> abs(1-0)=1 <= 1 --> return height = 2
- Root 1:   left=2, right=0  --> abs(2-0)=2 > 1  --> return -1 (Unbalanced!)

Result: checkHeight(root) = -1 != -1 --> false (Unbalanced)
```

---

### The Architectural Mental Model

```text
                          Balanced Binary Tree
                                    ↓
                              checkHeight(node)
                                    ↓
                             node == nullptr ?
                                 /     \
                           YES  /       \  NO
                               ↓         ↓
                           return 0    leftHeight = checkHeight(node->left)
                                       if (leftHeight == -1) return -1
                                                 ↓
                                       rightHeight = checkHeight(node->right)
                                       if (rightHeight == -1) return -1
                                                 ↓
                                       |leftHeight - rightHeight| > 1 ?
                                           /          \
                                     YES  /            \  NO
                                         ↓              ↓
                                     return -1     return 1 + max(left, right)
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** Sentinel return values (e.g., `-1`) allow combining subtree measurement with condition validation in a single bottom-up post-order pass, providing immediate short-circuiting upon invariant violation.
