# 1448. Count Good Nodes in Binary Tree

- **Problem Link:** https://leetcode.com/problems/count-good-nodes-in-binary-tree/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Trees`
- **LeetCode Topics:** `Tree` / `Depth-First Search` / `Breadth-First Search` / `Binary Tree`
- **Core Pattern:** `Depth-First Search (DFS) / Path State Propagation (Running Maximum)`
- **Last Practiced:** 2026-10-06
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given a binary tree `root`, a node $X$ in the tree is named **good** if in the path from the root to $X$ there are no nodes with a value greater than $X$.
  - Return the total number of **good** nodes in the binary tree.
  - Constraints: The number of nodes is in $[1, 10^5]$, and node values are in $[-10^4, 10^4]$.
- **The Core Challenge (Path History vs. Running State):**
  - A naive formulation would record every node along the root-to-current path into an array or stack (`vector<int> path`), checking whether $\text{node}->\text{val} \ge \max(\text{path})$.
  - Storing or copying explicit paths introduces unnecessary $O(H)$ time and space overhead per node.
- **The Path Compression Invariant:**
  - The definition of a "good node" asserts that:
    $$\text{node}->\text{val} \ge \max_{u \in \text{path}(\text{root}, \text{node})} u.\text{val}$$
  - All preceding nodes along the ancestral trajectory can be summarized cleanly into a single scalar:
    $$\text{maxValue} = \text{maximum value seen so far on the path from root to current node}$$
  - Once we check whether the current node satisfies $\text{node}->\text{val} \ge \text{maxValue}$, we update:
    $$\text{maxValue} = \max(\text{maxValue}, \text{node}->\text{val})$$
    and propagate this scalar downward to both children.
- **Bottom-Up Composition without Global State:**
  - Rather than mutating an external accumulator or global counter, each recursive DFS call returns the count of good nodes in its subtree:
    $$\text{total} = \text{good} + \text{dfs}(\text{node}->\text{left}, \text{maxValue}) + \text{dfs}(\text{node}->\text{right}, \text{maxValue})$$
  - This preserves functional purity and keeps recursion modular and re-entrant.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Explicit Path Tracking ($O(N \cdot H)$ time, $O(H)$ space):**
   - Push nodes onto a path vector during DFS, scanning the vector at each node to find the maximum.
   - *Drawback:* Expensive re-scanning degrades runtime to $O(N \cdot H)$, completely unnecessary.

2. **Approach 2 — BFS with Queue of Pairs ($O(N)$ time, $O(W)$ space):**
   - Push `pair<TreeNode*, int>` storing `(node, maxValue)` into a FIFO queue.
   - *Trade-off:* Perfectly valid, but requires heap allocation for queue nodes and pair packaging. BFS is unnecessary since no level-by-level boundaries or shortest-path guarantees are needed.

3. **Approach 3 — Top-Down DFS with Path Maximum Propagation (Chosen Optimal Solution):**
   - Initiate DFS from `root` with `maxValue = root->val`.
   - Base case: `!node` returns 0.
   - Check if current node is good: `int good = (node->val >= maxValue) ? 1 : 0`.
   - Update running maximum: `maxValue = max(maxValue, node->val)`.
   - Recurse into left and right subtrees with the updated `maxValue`, summing their returned counts.
   - *Verdict:* Optimal $O(N)$ time and $O(H)$ auxiliary space. Clean, self-contained, and naturally mirrors tree structural induction.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(N)$
  - Where $N$ is the number of nodes in the binary tree.
  - Each node is visited exactly once. At each node, comparison, max evaluation, and arithmetic take $O(1)$ constant time.
- **Space Complexity:** $O(H)$
  - Where $H$ is the height of the tree.
  - Governed strictly by the recursive call stack depth.
  - Balanced tree: $O(\log N)$ stack frames.
  - Degenerate (skewed) tree: $O(N)$ stack frames in the worst case.
  - No auxiliary heap allocations or collection containers are allocated.

---

## 4. Edge Cases & Gotchas

- [x] **Root Node is Always Good:** The root has no predecessors on its path; by definition, $\text{root}->\text{val} \ge \text{root}->\text{val}$, evaluating to `good = 1`. Calling `dfs(root, root->val)` guarantees this invariant.
- [x] **Evaluation Order Invariant:** Crucially, we must evaluate `node->val >= maxValue` **before** updating `maxValue = max(maxValue, node->val)`. Updating prematurely would cause every node to trivially equal `maxValue` and incorrectly be counted as good.
- [x] **Negative Values:** Node values range down to $-10^4$. Initializing `maxValue` with `root->val` avoids arbitrary sentinel initialization bugs (e.g., using 0 as a default would break all-negative trees).
- [x] **Duplicate Values on Path:** If multiple identical values appear along a path, all subsequent matching nodes are valid (`>=`), correctly incrementing `good`.

---

## 5. Clean Code

```cpp
#include <algorithm>

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
    int dfs(TreeNode* node, int maxValue) {
        if (!node) {
            return 0;
        }

        // A node is good if its value is greater than or equal to 
        // the maximum value seen on the root-to-node path.
        int good = node->val >= maxValue ? 1 : 0;
        maxValue = max(maxValue, node->val);

        return good + dfs(node->left, maxValue) + dfs(node->right, maxValue);
    }

public:
    int goodNodes(TreeNode* root) {
        return dfs(root, root->val);
    }
};
```

---

## 6. Visual Walkthrough & Architectural Pattern

### Step-by-Step Execution Trace

Given tree:
```text
        3
       / \
      1   4
     /   / \
    3   1   5
```

```text
Call dfs(root=3, maxValue=3):
├── Node 3: val=3 >= max=3 -> good=1, update max=3
│   ├── Left child: dfs(node=1, maxValue=3)
│   │   ├── Node 1: val=1 >= max=3 -> good=0, max remains 3
│   │   │   ├── Left child: dfs(node=3, maxValue=3)
│   │   │   │   └── Node 3: val=3 >= max=3 -> good=1, max remains 3
│   │   │   │       └── Children null -> returns 1
│   │   │   └── Right child: null -> returns 0
│   │   │   └── Subtree total = 0 + 1 + 0 = 1
│   │   └── Subtree total = 1
│   └── Right child: dfs(node=4, maxValue=3)
│       ├── Node 4: val=4 >= max=3 -> good=1, update max=4
│       │   ├── Left child: dfs(node=1, maxValue=4)
│       │   │   └── Node 1: val=1 >= max=4 -> good=0, max remains 4
│       │   │       └── Children null -> returns 0
│       │   └── Right child: dfs(node=5, maxValue=4)
│       │       └── Node 5: val=5 >= max=4 -> good=1, update max=5
│       │           └── Children null -> returns 1
│       │   └── Subtree total = 1 + 0 + 1 = 2
│       └── Subtree total = 2
└── Total Good Nodes = 1 (root) + 1 (left) + 2 (right) = 4
```

---

### The Architectural Mental Model

```text
                      Top-Down Path State Propagation
                                     ↓
                          dfs(node, maxValue)
                                     ↓
                               node == null? ─── Yes ───> return 0
                                     ↓ No
                        good = (node->val >= maxValue)
                                     ↓
                        maxValue = max(maxValue, node->val)
                                     ↓
                 ┌───────────────────┴───────────────────┐
                 ↓                                       ↓
     dfs(node->left, maxValue)               dfs(node->right, maxValue)
                 └───────────────────┬───────────────────┘
                                     ↓
                    return good + leftCount + rightCount
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** Compressing the root-to-node path history into a single downward-propagated scalar `maxValue` avoids tracking full paths, yielding an elegant $O(N)$ time and $O(H)$ space traversal.
