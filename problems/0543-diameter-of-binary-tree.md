# 0543. Diameter of Binary Tree

- **Problem Link:** https://leetcode.com/problems/diameter-of-binary-tree/
- **Difficulty:** `Easy`
- **NeetCode Category:** `Trees`
- **LeetCode Topics:** `Tree` / `Depth-First Search` / `Binary Tree`
- **Core Pattern:** `Post-Order Recursive DFS / Bottom-Up Height Return with Global Diameter Update`
- **Last Practiced:** 2026-10-04
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given the `root` of a binary tree, return the length of the **diameter** of the tree.
  - The diameter is the length of the **longest path between any two nodes** in a tree.
  - This path **may or may not pass through the root**.
  - The length of a path between two nodes is represented by the **number of edges** between them.
  - Constraints: Number of nodes in $[1, 10^4]$, node values in $[-100, 100]$.
- **Core intuition:**
  - Any path in a binary tree has a unique highest node (its Lowest Common Ancestor or "turning point").
  - If a node is the turning point of a path, the longest downward branch into its left subtree has length `leftHeight`, and the longest downward branch into its right subtree has length `rightHeight`.
  - The total number of edges along this turning path is precisely:
    $$\text{pathLength}(node) = \text{height}(node.left) + \text{height}(node.right)$$
  - Because the global diameter could bend at *any* node (not necessarily the root), we must evaluate this metric across all nodes and maintain the global maximum.
- **Why Post-Order Traversal is Mandatory:**
  - To evaluate the path length at `node`, we strictly require the heights of both its left and right subtrees beforehand.
  - This demands a bottom-up **post-order traversal** (`left -> right -> root`): solve subproblems on children first, then synthesize and propagate results back up to the parent.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Naive Top-Down Traversal ($O(N^2)$ time, $O(h)$ space):**
   - For every node, make independent top-down calls to a helper `height(node->left)` and `height(node->right)`.
   - Update `maxDiameter` with `leftHeight + rightHeight`, then recursively repeat for `node->left` and `node->right`.
   - *Drawback:* Repeatedly recalculates heights of identical subtrees. On degenerate (skewed) trees, this degenerates into quadratic $O(N^2)$ time complexity.

2. **Approach 2 — Bottom-Up DFS Returning Pairs ($O(N)$ time, $O(h)$ space):**
   - Each recursive call returns a pair/tuple `pair<int, int>` representing `{subtree_height, max_diameter_in_subtree}`.
   - *Drawback:* Pure functional design avoids mutable state, but introduces object instantiation and unpacking overhead across $N$ recursive call frames.

3. **Approach 3 — Single-Pass Post-Order DFS with Member Variable (Chosen Optimal Solution):**
   - Maintain a private member variable `diameter = 0`.
   - The recursive function `height(node)` returns the height of the rooted subtree (number of nodes along the deepest path, defined as `1 + max(leftHeight, rightHeight)`).
   - Before returning, it updates the global maximum in-place:
     $$\text{diameter} = \max(\text{diameter}, \text{leftHeight} + \text{rightHeight})$$
   - *Verdict:* Optimal $O(N)$ time and $O(h)$ space. Every node is visited exactly once, computing height and updating the diameter concurrently with zero redundant traversals or allocation overhead.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(N)$
  - Where $N$ is the total number of nodes in the binary tree.
  - Every node is visited exactly once during the post-order DFS.
  - At each node, computing the sum and maximum operations takes constant $O(1)$ time.
- **Space Complexity:** $O(h)$
  - Where $h$ represents the height of the binary tree, corresponding to the maximum depth of the call stack.
  - **Balanced Tree:** $h = O(\log N)$ frames.
  - **Degenerate / Skewed Tree:** $h = O(N)$ frames.
  - Auxiliary heap allocations are strictly $O(1)$.

---

## 4. Edge Cases & Gotchas

- [x] **Path Not Passing Through the Root:** The diameter may reside entirely within a deep subtree (e.g., when the left subtree is broad and bushy while the right subtree is shallow). Updating the global `diameter` at every node guarantees capturing this subtree-local optimum.
- [x] **Single-Node Tree (`root->left == root->right == nullptr`):** Both children return height `0`. The diameter is `max(0, 0 + 0) = 0`, and the function returns `0` edges. Correct, as a single node has zero connecting edges.
- [x] **Counting Edges vs. Nodes:** The problem measures diameter by the number of edges, not nodes. If a path contains $k$ nodes, it has $k - 1$ edges. Since `height` returns node count from leaf ($1$ for a leaf), `leftHeight + rightHeight` exactly equals the edge count connecting the deepest leaves on both sides through the current node (e.g., for a node with two leaf children, `1 + 1 = 2` edges across 3 nodes). No $+1$ offset is needed.

---

## 5. Clean Code (Optimal Solution: Post-Order Recursive DFS)

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
    int diameter = 0;

    int height(TreeNode* node) {
        if (!node) {
            return 0;
        }

        int leftHeight = height(node->left);
        int rightHeight = height(node->right);

        // Longest path (in edges) turning at this node.
        diameter = max(diameter, leftHeight + rightHeight);

        // Return height of the current subtree (in nodes) to parent.
        return 1 + max(leftHeight, rightHeight);
    }

public:
    int diameterOfBinaryTree(TreeNode* root) {
        height(root);
        return diameter;
    }
};
```

---

## 6. Review & Takeaways

### Visual Dry-Run Tracing

```text
Given Binary Tree:
          1
        /   \
       2     3
      / \
     4   5

Step-by-Step Bottom-Up Evaluation:

1. Leaf 4:
   - leftHeight = 0, rightHeight = 0
   - local diameter = 0 + 0 = 0 -> diameter = max(0, 0) = 0
   - return height = 1 + max(0, 0) = 1

2. Leaf 5:
   - leftHeight = 0, rightHeight = 0
   - local diameter = 0 + 0 = 0 -> diameter = max(0, 0) = 0
   - return height = 1 + max(0, 0) = 1

3. Node 2:
   - leftHeight = 1 (from 4), rightHeight = 1 (from 5)
   - local diameter = 1 + 1 = 2 (path: 4 - 2 - 5)
   - diameter = max(0, 2) = 2
   - return height = 1 + max(1, 1) = 2

4. Leaf 3:
   - leftHeight = 0, rightHeight = 0
   - local diameter = 0 + 0 = 0 -> diameter = max(2, 0) = 2
   - return height = 1 + max(0, 0) = 1

5. Root 1:
   - leftHeight = 2 (from 2), rightHeight = 1 (from 3)
   - local diameter = 2 + 1 = 3 (path: 4 - 2 - 1 - 3 or 5 - 2 - 1 - 3)
   - diameter = max(2, 3) = 3
   - return height = 1 + max(2, 1) = 3

Final Result: diameter = 3 (edges)
```

---

### The Architectural Mental Model

```text
                    Bottom-Up Post-Order Traversal Pattern
                                      ↓
                                 Current Node
                                   /       \
                         (Recurse) /         \ (Recurse)
                                  ↓           ↓
                             leftHeight   rightHeight
                                  \           /
                                   \         /
                                        ↓
                       1. Combine Child Metrics:
                          localDiameter = leftHeight + rightHeight
                                        ↓
                       2. Update Global Accumulator:
                          diameter = max(diameter, localDiameter)
                                        ↓
                       3. Return Subtree Metric to Ancestor:
                          return 1 + max(leftHeight, rightHeight)
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** Whenever an optimal tree path can turn at any node, use bottom-up post-order DFS: return subtree height upward while simultaneously updating the global path maximum in-place.
