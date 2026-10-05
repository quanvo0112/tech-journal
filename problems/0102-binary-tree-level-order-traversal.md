# 0102. Binary Tree Level Order Traversal

- **Problem Link:** https://leetcode.com/problems/binary-tree-level-order-traversal/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Trees`
- **LeetCode Topics:** `Tree` / `Breadth-First Search` / `Binary Tree`
- **Core Pattern:** `Breadth-First Search (BFS) / Level-Order Chunking with Queue Size Snapshot`
- **Last Practiced:** 2026-10-05
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given the `root` of a binary tree, return the **level order traversal** of its nodes' values (i.e., from left to right, level by level).
  - Return type is `vector<vector<int>>`, requiring values at each depth level to be partitioned into separate sub-arrays.
  - Constraints: Number of nodes in $[0, 2000]$, node values in $[-1000, 1000]$.
- **The Core Challenge (Chunking into Levels):**
  - A standard Breadth-First Search (BFS) with a First-In-First-Out (FIFO) queue naturally visits nodes level by level, but flattens them into an undifferentiated 1D stream.
  - To group elements into distinct 2D level buckets without confusing the boundaries between generations, we need a delimiter.
- **The Invariant & Snapshot Technique:**
  - At the beginning of each while-loop iteration, the queue contains **exclusively** all nodes belonging to the current depth level.
  - By taking an instantaneous snapshot:
    $$\text{levelSize} = q.\text{size}()$$
  - We run an inner loop exactly `levelSize` times. Any child nodes pushed into the queue during this processing belong strictly to the *subsequent* level and are buffered safely behind the current generation.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Preorder DFS with Depth Index ($O(N)$ time, $O(h)$ space):**
   - Traverse the tree recursively via Preorder DFS, passing `depth` as an argument.
   - If `depth == result.size()`, allocate a new sub-vector `result.push_back({})`. Append `result[depth].push_back(node->val)`.
   - *Trade-off:* Valid and memory-efficient on skewed trees ($O(h)$ space), but relies on arbitrary index addressing rather than true breadth-first exploration.

2. **Approach 2 — BFS with Sentinel / Pair Queue ($O(N)$ time, $O(W)$ space):**
   - Store pairs `queue<pair<TreeNode*, int>>` storing depth explicitly, or inject `nullptr` sentinel markers at level boundaries.
   - *Drawback:* Adds extra allocation overhead or complex conditional branching per dequeued element.

3. **Approach 3 — Iterative BFS with Queue Size Snapshot (Chosen Optimal Solution):**
   - Initialize `queue<TreeNode*> q` and push `root`.
   - While `!q.empty()`:
     1. Read `levelSize = q.size()`.
     2. Allocate an empty `vector<int> level`.
     3. Pop exactly `levelSize` nodes, append their values to `level`, and enqueue their non-null children (`node->left`, `node->right`).
     4. Append `level` to `result`.
   - *Verdict:* Optimal $O(N)$ time and $O(W)$ space. Canonical, idiomatic BFS level-order pattern with minimal boilerplate.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(N)$
  - Where $N$ is the total number of nodes in the binary tree.
  - Every node is enqueued and dequeued exactly once.
  - Inner vector operations take constant amortized $O(1)$ time per node.
- **Space Complexity:** $O(W)$
  - Where $W$ is the maximum width of the binary tree (the maximum number of nodes residing at any single level).
  - In a complete or full binary tree, the leaf level contains up to $\lceil N / 2 \rceil$ nodes, leading to worst-case queue space of $O(N)$.
  - On a degenerate / skewed tree (linked-list-like), $W = 1$, yielding $O(1)$ queue memory.
  - The return container `result` stores all $N$ node values.

---

## 4. Edge Cases & Gotchas

- [x] **Empty Tree (`root == nullptr`):** Handled immediately by `if (!root) return {};` returning an empty 2D vector without touching the queue.
- [x] **Single-Node Tree:** Queue processes size `1`, pops `root`, discovers no children, and returns `[[root->val]]`.
- [x] **Evaluating `q.size()` Inside the Loop Condition (Critical Gotcha):**
  - **Fatal Bug:** Writing `for (int i = 0; i < q.size(); ++i)` recalculates `q.size()` dynamically as children are pushed, causing the loop to bleed across multiple levels.
  - **Correct Invariant:** Freeze `int levelSize = q.size();` into a local variable *before* entering the inner loop.

---

## 5. Clean Code (Optimal Solution: BFS with Queue Size Snapshot)

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
    vector<vector<int>> levelOrder(TreeNode* root) {
        if (!root) {
            return {};
        }

        vector<vector<int>> result;
        queue<TreeNode*> q;

        q.push(root);

        while (!q.empty()) {
            // Snapshot the number of nodes belonging strictly to the current level
            int levelSize = q.size();
            vector<int> level;

            for (int i = 0; i < levelSize; ++i) {
                TreeNode* node = q.front();
                q.pop();

                level.push_back(node->val);

                if (node->left) {
                    q.push(node->left);
                }
                if (node->right) {
                    q.push(node->right);
                }
            }

            result.push_back(level);
        }

        return result;
    }
};
```

---

## 6. Review & Takeaways

### Visual BFS Level Execution Trace

```text
Given Binary Tree:
        3
       / \
      9  20
         / \
        15  7

Iteration 0 (Level 0):
- Queue state:   [3]
- levelSize:     1
- Process:       Pop 3 -> level = [3]; Push 9, 20
- Queue after:   [9, 20]
- Result:        [[3]]

Iteration 1 (Level 1):
- Queue state:   [9, 20]
- levelSize:     2
- Process:       Pop 9  -> level = [9]; no children
                 Pop 20 -> level = [9, 20]; Push 15, 7
- Queue after:   [15, 7]
- Result:        [[3], [9, 20]]

Iteration 2 (Level 2):
- Queue state:   [15, 7]
- levelSize:     2
- Process:       Pop 15 -> level = [15]; no children
                 Pop 7  -> level = [15, 7]; no children
- Queue after:   []
- Result:        [[3], [9, 20], [15, 7]]

Queue empty -> Terminate and return result.
```

---

### The Architectural Mental Model

```text
                     Level-Order Traversal (BFS)
                                  ↓
                        queue.push(root)
                                  ↓
                        while (!queue.empty())
                                  ↓
                     levelSize = queue.size()  <─── Snapshot Level Count
                                  ↓
                      for i from 0 to levelSize:
                      ┌──────────────────────┐
                      │  node = queue.pop()  │
                      │  level.push(val)     │
                      │  enqueue children    │
                      └──────────────────────┘
                                  ↓
                       result.push(level)
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** Freezing `int levelSize = q.size()` before processing creates a deterministic barrier between generations in a standard FIFO queue, enabling clean $\mathcal{O}(N)$ level-by-level chunking.
