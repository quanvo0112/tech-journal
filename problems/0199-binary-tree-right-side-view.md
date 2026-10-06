# 0199. Binary Tree Right Side View

- **Problem Link:** https://leetcode.com/problems/binary-tree-right-side-view/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Trees`
- **LeetCode Topics:** `Tree` / `Depth-First Search` / `Breadth-First Search` / `Binary Tree`
- **Core Pattern:** `Right-First Recursive DFS / Depth Invariant Matching`
- **Last Practiced:** 2026-10-06
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given the `root` of a binary tree, imagine yourself standing on the **right side** of it.
  - Return the values of the nodes you can see ordered from top to bottom.
  - Number of nodes in $[0, 100]$, node values in $[-100, 100]$.
- **The Core Insight:**
  - At each level/depth $d$, exactly **one** node is visible from the right: the rightmost node at depth $d$.
  - A node is visible if and only if no other node exists to its right at that exact depth.
  - If a level has both left and right subtrees, the right child obscures the left child. However, if the right subtree is shorter or missing at depth $d$, a node from the left subtree becomes visible.
- **The Invariant Formulation:**
  - If we traverse the tree prioritizing the right branch before the left branch (`Root -> Right -> Left`), the **first node we encounter at any depth $d$** is guaranteed to be the rightmost node at that level.
  - Since `result` collects one node per depth starting from $0$, the condition:
    $$\text{depth} == \text{result.size}()$$
    holds true if and only if we have not yet recorded any node for the current `depth`.
  - When this condition matches, we push `node->val` into `result`. Any subsequent nodes visited at that same depth will find $\text{depth} < \text{result.size}()$ and will be skipped.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Level-Order Traversal (BFS with Queue) ($O(N)$ time, $O(W)$ space):**
   - Traverse the tree level by level using a queue with snapshot sizing (`levelSize = q.size()`).
   - For each level, append the value of the last node (`i == levelSize - 1`) to the output.
   - *Trade-off:* Highly intuitive and straightforward, but requires maintaining an explicit queue data structure holding up to $O(W)$ nodes (up to $N/2$ nodes on complete binary trees).
2. **Approach 2 — Right-First Recursive DFS (Chosen Optimal Solution):**
   - Perform a Depth-First Search in `Root -> Right -> Left` order, keeping track of the current `depth` (starting at 0).
   - If `depth == result.size()`, append `node->val` to `result`.
   - Recursively call `dfs(node->right, depth + 1)` first, followed by `dfs(node->left, depth + 1)`.
   - *Verdict:* Optimal $O(N)$ time and $O(H)$ auxiliary space (recursion stack). Extremely elegant, minimal boilerplate, eliminates explicit queue allocations, and directly encodes the right-priority visibility invariant.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(N)$
  - Where $N$ is the total number of nodes in the binary tree.
  - Each node is visited exactly once during the recursive DFS traversal.
- **Space Complexity:** $O(H)$
  - Where $H$ is the height of the binary tree.
  - Auxiliary space is dictated strictly by the recursion call stack.
  - In a balanced tree, $H = O(\log N)$. In the worst-case degenerate (skewed) tree, $H = O(N)$.
  - $O(H)$ space is superior or equal to BFS's $O(W)$ space on wide, balanced trees ($O(\log N)$ vs $O(N)$).

---

## 4. Edge Cases & Gotchas

- [x] **Empty Tree (`root == nullptr`):** Handled gracefully by returning an empty vector `{}` immediately (the base condition `!node` terminates traversal).
- [x] **Single Node Tree:** Returns `[root->val]` with depth 0.
- [x] **Left-Heavy Tree (No Right Subtree):** The right subtree is empty, so DFS falls through to the left subtree. Nodes from the left branch are correctly selected because `depth == result.size()` triggers for them.
- [x] **Right Subtree Shorter Than Left Subtree:** Upper levels show nodes from the right subtree; deeper levels uncover and record nodes from the left subtree once the right subtree terminates.

---

## 5. Clean Code

```cpp
#include <vector>

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
    vector<int> result;

    void dfs(TreeNode* node, int depth) {
        if (!node) {
            return;
        }

        // First node encountered at this depth is the rightmost visible node.
        if (depth == result.size()) {
            result.push_back(node->val);
        }

        // Prioritize right child over left child.
        dfs(node->right, depth + 1);
        dfs(node->left, depth + 1);
    }

public:
    vector<int> rightSideView(TreeNode* root) {
        result.clear();
        dfs(root, 0);
        return result;
    }
};
```

---

## 6. Visual Walkthrough & Architectural Pattern

### Step-by-Step Execution Trace

Given tree:
```text
        1
       / \
      2   3
       \   \
        5   4
```

1. **Initial Call:** `dfs(node=1, depth=0)`.
   - `depth (0) == result.size() (0)` $\rightarrow$ append `1`. `result = [1]`.
   - Recurse right: `dfs(node=3, depth=1)`.

2. **Node 3:**
   - `depth (1) == result.size() (1)` $\rightarrow$ append `3`. `result = [1, 3]`.
   - Recurse right: `dfs(node=4, depth=2)`.

3. **Node 4:**
   - `depth (2) == result.size() (2)` $\rightarrow$ append `4`. `result = [1, 3, 4]`.
   - Recurse right: `nullptr`, return.
   - Recurse left: `nullptr`, return.
   - Unwind back to Node 3 $\rightarrow$ Recurse left: `nullptr`, return.
   - Unwind back to Node 1 $\rightarrow$ Recurse left: `dfs(node=2, depth=1)`.

4. **Node 2:**
   - `depth (1) < result.size() (3)` $\rightarrow$ do not append (already saw Node 3 at depth 1).
   - Recurse right: `dfs(node=5, depth=2)`.

5. **Node 5:**
   - `depth (2) < result.size() (3)` $\rightarrow$ do not append (already saw Node 4 at depth 2).
   - Recurse right: `nullptr`, return.
   - Recurse left: `nullptr`, return.

6. **Final Result:** `[1, 3, 4]`.

---

### The Architectural Mental Model

```text
                     Right-First DFS Traversal
                                 ↓
                          dfs(node, depth)
                                 ↓
                           node == null? ─── Yes ───> return
                                 ↓ No
                    depth == result.size()?
                                 │
                 ┌───────────────┴───────────────┐
                 │ Yes                           │ No
                 ↓                               ↓
       result.push(node->val)            (Skip recording: right
                 │                        node already captured)
                 └───────────────┬───────────────┘
                                 ↓
                     dfs(node->right, depth + 1)  <── Priority 1
                                 ↓
                     dfs(node->left, depth + 1)   <── Priority 2
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** By traversing `Root -> Right -> Left`, the first node visited at each depth is guaranteed to be the rightmost node visible from the side.
