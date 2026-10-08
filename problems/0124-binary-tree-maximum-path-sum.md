# 0124. Binary Tree Maximum Path Sum

- **Problem Link:** https://leetcode.com/problems/binary-tree-maximum-path-sum/
- **Difficulty:** `Hard`
- **NeetCode Category:** `Trees`
- **LeetCode Topics:** `Dynamic Programming` / `Tree` / `Depth-First Search` / `Binary Tree`
- **Core Pattern:** `Postorder DFS / Dual Path Semantics (Inverted-V Summit vs Extensible Branch Gain) with Global Maximizer`
- **Last Practiced:** 2026-10-07
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - A **path** in a binary tree is a sequence of nodes where each pair of adjacent nodes has an edge connecting them. A node can only appear in the sequence at most once.
  - The path does **not** need to pass through the root, and may start and finish at any arbitrary node.
  - Return the maximum **path sum** of any non-empty path.
  - Constraints: The number of nodes is in $[1, 3 \cdot 10^4]$, and node values are in $[-1000, 1000]$.
- **The Core Dilemma (Forking vs. Extending):**
  - A path can turn through a node $u$ by combining its left subtree, node $u$, and its right subtree ($\text{Left} \to u \to \text{Right}$). This forms an **inverted-V ($\land$) summit**.
  - However, such a path **cannot** be extended upward to $u$'s parent. A simple path cannot branch into both children and simultaneously continue upward to an ancestor without visiting node $u$ twice.
- **The Dual-Value Invariant:**
  - At every node $u$, we must disentangle two distinct path concepts:
    1. **Locally Closed Summit Path (`currentPath`):**
       $$\text{currentPath} = u.\text{val} + \text{leftGain} + \text{rightGain}$$
       This path peaks at $u$ and terminates within $u$'s subtree. It represents a potential complete candidate for the global answer:
       $$\text{maxSum} = \max(\text{maxSum}, \text{currentPath})$$
    2. **Upward-Extensible Branch Gain (`maxGain`):**
       $$\text{return } u.\text{val} + \max(\text{leftGain}, \text{rightGain})$$
       This is the maximum linear chain passing through $u$ that can be concatenated to $u$'s parent. The parent can extend into at most **one** child branch.
- **Negative Subtree Pruning (`max(0, ...)`):**
  - If a child subtree produces a net negative gain ($\text{gain} < 0$), including it strictly reduces any path sum.
  - We greedily prune negative branches by clamping:
    $$\text{leftGain} = \max(0, \text{maxGain}(u.\text{left})), \quad \text{rightGain} = \max(0, \text{maxGain}(u.\text{right}))$$
  - Clamping to 0 corresponds to omitting the entire subtree from the path.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Brute Force Downward DFS ($O(N^2)$ time, $O(H)$ space):**
   - For every node, compute the max downward path sum into the left and right subtrees independently.
   - *Drawback:* Repeatedly traverses subtrees from scratch, degrading to $O(N^2)$ runtime on skewed trees.

2. **Approach 2 — Postorder DFS with Bottom-Up Dynamic Programming (Chosen Optimal Solution):**
   - Traverse the tree via Postorder DFS (`Left -> Right -> Root`).
   - For each node:
     1. Recursively compute clamped gains from left and right children: $\max(0, \text{maxGain}(\text{child}))$.
     2. Evaluate the inverted-V summit sum passing through the node: $u.\text{val} + \text{leftGain} + \text{rightGain}$.
     3. Update the global variable $\text{maxSum} = \max(\text{maxSum}, \text{currentPath})$.
     4. Return the maximum single branch sum extendable to the parent: $u.\text{val} + \max(\text{leftGain}, \text{rightGain})$.
   - *Verdict:* Optimal $O(N)$ time and $O(H)$ space. Computes the exact global maximum in a single bottom-up pass with zero redundant recalculation. Mirror pattern to LeetCode 0543 (Diameter of Binary Tree), generalized to weighted path sums.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(N)$
  - Where $N$ is the number of nodes in the binary tree.
  - Every node is visited exactly once in the postorder traversal.
  - Constant $O(1)$ operations (clamping, additions, comparisons) are performed per node.
- **Space Complexity:** $O(H)$
  - Where $H$ is the height of the binary tree.
  - Governed strictly by the recursion call stack.
  - Balanced tree: $O(\log N)$ stack frames.
  - Degenerate (skewed) tree: $O(N)$ stack frames in the worst case.
  - Auxiliary memory overhead is $O(1)$.

---

## 4. Edge Cases & Gotchas

- [x] **All-Negative Values (e.g. `[-5]`, `[-3, -2, -1]`):**
  - Initializing `maxSum = 0` is a fatal bug because the problem requires a non-empty path. An all-negative tree must return the largest single negative element (e.g. `-1`), not `0`.
  - Initializing `maxSum = INT_MIN` ensures correct tracking. Even though child gains clamp to $0$, the current node value is added without clamping:
    $$\text{currentPath} = \text{node}->\text{val} + 0 + 0 = \text{node}->\text{val}$$
    Thus, single negative nodes are safely captured in `maxSum`.
- [x] **Non-Root Maximum Path:** The optimal path frequently does not pass through the tree's root. For example, in `[-10, 9, 20, null, null, 15, 7]`, the summit path at node `20` yields $15 + 20 + 7 = 42$, whereas routing through root `-10` degrades the sum to $41$.
- [x] **Single-Node Tree:** Returns `root->val` directly.

---

## 5. Clean Code

```cpp
#include <algorithm>
#include <climits>

using namespace std;

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
    int maxSum = INT_MIN;

    int maxGain(TreeNode* node) {
        if (!node) {
            return 0;
        }

        // Clamp negative child contributions to 0 (prune from path)
        int leftGain = max(0, maxGain(node->left));
        int rightGain = max(0, maxGain(node->right));

        // Best path with the current node serving as the summit (inverted-V)
        int currentPath = node->val + leftGain + rightGain;
        maxSum = max(maxSum, currentPath);

        // Maximum single branch gain extendable to the parent
        return node->val + max(leftGain, rightGain);
    }

public:
    int maxPathSum(TreeNode* root) {
        maxSum = INT_MIN;
        maxGain(root);
        return maxSum;
    }
};
```

---

## 6. Visual Walkthrough & Architectural Pattern

### Step-by-Step Execution Trace

Given tree:
```text
         -10
         /  \
        9    20
            /  \
           15   7
```

1. **Node 9 (Leaf):**
   - `leftGain = 0`, `rightGain = 0`
   - `currentPath = 9 + 0 + 0 = 9` $\rightarrow$ `maxSum = 9`
   - Return to parent: $9 + 0 = 9$

2. **Node 15 (Leaf):**
   - `leftGain = 0`, `rightGain = 0`
   - `currentPath = 15 + 0 + 0 = 15` $\rightarrow$ `maxSum = 15`
   - Return to parent: $15 + 0 = 15$

3. **Node 7 (Leaf):**
   - `leftGain = 0`, `rightGain = 0`
   - `currentPath = 7 + 0 + 0 = 7` $\rightarrow$ `maxSum = 15`
   - Return to parent: $7 + 0 = 7$

4. **Node 20:**
   - `leftGain = max(0, 15) = 15`
   - `rightGain = max(0, 7) = 7`
   - `currentPath = 20 + 15 + 7 = 42` $\rightarrow$ `maxSum = max(15, 42) = 42` (Path: $15 \to 20 \to 7$)
   - Return to parent: $20 + \max(15, 7) = 35$

5. **Root Node -10:**
   - `leftGain = max(0, 9) = 9`
   - `rightGain = max(0, 35) = 35`
   - `currentPath = -10 + 9 + 35 = 34` $\rightarrow$ `maxSum = max(42, 34) = 42`
   - Return: $-10 + 35 = 25$

**Final Result:** `42` (Optimal path: $15 \to 20 \to 7$, completely bypassing the root).

---

### The Architectural Mental Model

```text
                                 Node u
                                 /    \
                     maxGain(left)    maxGain(right)
                           │                │
                           ▼                ▼
                 leftGain = max(0, L)  rightGain = max(0, R)
                           │                │
            ┌──────────────┴────────────────┴──────────────┐
            ▼                                              ▼
   Summit Path (Local Peak)                       Branch Extension (Return)
   u.val + leftGain + rightGain                   u.val + max(leftGain, rightGain)
            │                                              │
            ▼                                              ▼
   maxSum = max(maxSum, path)                     Sent upward to parent
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** Combine both child branches (`u + left + right`) to update the global summit answer, but extend only the single best branch (`u + max(left, right)`) upward to preserve valid simple path geometry.
