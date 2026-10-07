# 0105. Construct Binary Tree from Preorder and Inorder Traversal

- **Problem Link:** https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Trees`
- **LeetCode Topics:** `Array` / `Hash Table` / `Divide and Conquer` / `Tree` / `Binary Tree`
- **Core Pattern:** `Hash Map + Recursive Divide & Conquer with Sequential Preorder Pointer`
- **Last Practiced:** 2026-10-07
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given two integer arrays `preorder` and `inorder` where `preorder` is the preorder traversal of a binary tree and `inorder` is the inorder traversal of the same tree, construct and return the binary tree.
  - Constraints: $1 \le \text{length} \le 3000$, all values are strictly **unique**, and `inorder` is guaranteed to be the valid inorder traversal for `preorder`.
- **The Complementary Properties of Traversals:**
  1. **Preorder (`Root -> Left -> Right`):**
     - The very first element in any preorder slice is definitively the **root** of that subtree.
  2. **Inorder (`Left -> Root -> Right`):**
     - Finding the root element splits the array into two disjoint segments:
       $$\text{inorder} = [\underbrace{\text{Left Subtree}}_{0 \dots \text{mid}-1}] \quad \text{Root}_{\text{mid}} \quad [\underbrace{\text{Right Subtree}}_{\text{mid}+1 \dots \text{end}}]$$
- **The Core Divide-and-Conquer Mental Model:**
  - `preorder` identifies **who** the root node is.
  - `inorder` determines **how** the remaining nodes are split between the left and right subtrees.
- **Why Naive Search is $O(N^2)$ and Why Hash Map is $O(N)$:**
  - Searching for `rootValue` linearly inside `inorder` takes $O(N)$ at each recursive step, compounding to $O(N^2)$ in unbalanced or degenerate trees.
  - Precomputing an `unordered_map<int, int> inorderIndex` before recursion grants $O(1)$ average lookup time for the split index `mid`.
- **Sequential Pointer Optimization (`preorderIndex`):**
  - Because preorder visits `Root -> Left -> Right`, a single monotonic index `int preorderIndex = 0` advanced via `preorder[preorderIndex++]` visits roots in their exact construction sequence.
  - As long as we recursively build `root->left` before `root->right`, `preorderIndex` naturally advances from the current root into the left subtree's root, and upon left completion, stands exactly at the right subtree's root—eliminating any need to calculate explicit `preorder` range indices.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Recursive Subarray Copying ($O(N^2)$ time, $O(N^2)$ space):**
   - Locate root in `inorder`, slice `inorder` and `preorder` into new sub-vectors, and pass copies recursively.
   - *Drawback:* Expensive vector allocations and copying at every stack frame.

2. **Approach 2 — Index Ranges with Linear Scan ($O(N^2)$ time, $O(H)$ space):**
   - Pass indices `[left, right]` without vector copying, but use `std::find()` on `inorder` to locate `mid`.
   - *Drawback:* $O(N)$ per node lookup causes worst-case $O(N^2)$ time complexity.

3. **Approach 3 — Hash Map + Inorder Boundaries + Sequential Preorder Pointer (Chosen Optimal Solution):**
   - Populate `inorderIndex[inorder[i]] = i` in $O(N)$ time.
   - Recursively call `build(left, right)`:
     - Base case: `if (left > right) return nullptr`.
     - Extract `rootValue = preorder[preorderIndex++]` and instantiate `new TreeNode(rootValue)`.
     - Locate `mid = inorderIndex[rootValue]`.
     - Recursively assign `root->left = build(left, mid - 1)`.
     - Recursively assign `root->right = build(mid + 1, right)`.
   - *Verdict:* Optimal $O(N)$ time and $O(N)$ space. Minimal memory footprint, zero redundant vector slicing, and canonical clean recursion.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(N)$
  - Building the hash map takes $O(N)$ time for $N$ insertions.
  - The recursive function `build` is invoked exactly $2N + 1$ times (once per node, plus null leaf children). Each invocation does $O(1)$ work (hash map lookup, pointer increment, pointer assignments).
  - Overall time complexity is strictly $O(N)$.
- **Space Complexity:** $O(N)$
  - Auxiliary space for the hash map `inorderIndex` storing $N$ key-value pairs is $O(N)$.
  - The recursion stack consumes $O(H)$ frames where $H$ is the tree height ($O(\log N)$ average for balanced trees, $O(N)$ worst case for skewed trees).
  - Total auxiliary space is $O(N)$.

---

## 4. Edge Cases & Gotchas

- [x] **Empty Range (`left > right`):** When a node has no left child (i.e. `mid == left`), the left recursive call receives `build(left, mid - 1)` which evaluates to `left > mid - 1`, returning `nullptr` cleanly.
- [x] **Strict Construction Order (`Left` then `Right`):** Because `preorderIndex` increments monotonically across recursive calls, we **must** call `build` on `root->left` before `root->right`. Inverting this order breaks the traversal sequence invariant.
- [x] **Unique Values Guarantee:** The problem guarantees unique values in both arrays, eliminating ambiguous duplicate keys in the hash map.
- [x] **Single-Node Tree:** Handled seamlessly with `left = 0, right = 0`, producing a single node with null children.

---

## 5. Clean Code

```cpp
#include <vector>
#include <unordered_map>

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
    unordered_map<int, int> inorderIndex;
    int preorderIndex = 0;

    TreeNode* build(vector<int>& preorder, int left, int right) {
        if (left > right) {
            return nullptr;
        }

        // Preorder: Root -> Left -> Right
        int rootValue = preorder[preorderIndex++];
        TreeNode* root = new TreeNode(rootValue);

        // Find root position in inorder to partition subtrees
        int mid = inorderIndex[rootValue];

        // Inorder: Left -> Root -> Right (left subtree must be built first)
        root->left = build(preorder, left, mid - 1);
        root->right = build(preorder, mid + 1, right);

        return root;
    }

public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        inorderIndex.clear();
        preorderIndex = 0;

        for (int i = 0; i < inorder.size(); ++i) {
            inorderIndex[inorder[i]] = i;
        }

        return build(preorder, 0, inorder.size() - 1);
    }
};
```

---

## 6. Visual Walkthrough & Architectural Pattern

### Step-by-Step Execution Trace

Given:
- `preorder = [3, 9, 20, 15, 7]`
- `inorder  = [9, 3, 15, 20, 7]`
- Hash Map: `{9: 0, 3: 1, 15: 2, 20: 3, 7: 4}`

```text
1. build(left=0, right=4):
   - rootValue = preorder[0] = 3 (preorderIndex -> 1)
   - mid = inorderIndex[3] = 1
   - Left subtree in inorder: [0, 0] -> [9]
   - Right subtree in inorder: [2, 4] -> [15, 20, 7]

2. build(left=0, right=0) [Left child of 3]:
   - rootValue = preorder[1] = 9 (preorderIndex -> 2)
   - mid = inorderIndex[9] = 0
   - build(0, -1) -> nullptr
   - build(1, 0)  -> nullptr
   - Node 9 returned as root->left.

3. build(left=2, right=4) [Right child of 3]:
   - rootValue = preorder[2] = 20 (preorderIndex -> 3)
   - mid = inorderIndex[20] = 3
   - Left subtree in inorder: [2, 2] -> [15]
   - Right subtree in inorder: [4, 4] -> [7]

4. build(left=2, right=2) [Left child of 20]:
   - rootValue = preorder[3] = 15 (preorderIndex -> 4)
   - mid = inorderIndex[15] = 2
   - Children null -> Node 15 returned.

5. build(left=4, right=4) [Right child of 20]:
   - rootValue = preorder[4] = 7 (preorderIndex -> 5)
   - mid = inorderIndex[7] = 4
   - Children null -> Node 7 returned.

Resulting Binary Tree:
        3
       / \
      9  20
        /  \
       15   7
```

---

### The Architectural Mental Model

```text
               Preorder Traversal              Inorder Traversal
               [Root, Left, Right]            [Left, Root, Right]
                       │                               │
                       ▼                               ▼
            preorder[preorderIndex++]       inorderIndex[rootValue]
                       │                               │
                 Identifies Root               Partitions Subtrees
                       └──────────────┬────────────────┘
                                      ▼
                        TreeNode* root = new TreeNode(val)
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
      root->left = build(left, mid - 1)     root->right = build(mid + 1, right)
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** Preorder provides root identities sequentially while inorder establishes subtree boundaries; coupling an $O(1)$ hash map lookup with a monotonic preorder pointer constructs the entire tree in optimal $O(N)$ time without array copying.

