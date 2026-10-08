# 0230. Kth Smallest Element in a BST

- **Problem Link:** https://leetcode.com/problems/kth-smallest-element-in-a-bst/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Trees`
- **LeetCode Topics:** `Tree` / `Depth-First Search` / `Binary Search Tree` / `Binary Tree`
- **Core Pattern:** `Iterative Inorder Traversal with Explicit Stack (Early Exit at k-th Node)`
- **Last Practiced:** 2026-10-08
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given the `root` of a binary search tree (BST) and an integer `k`, return the $k^{\text{th}}$ smallest value (1-indexed) of all the values of the nodes in the tree.
  - Constraints: $n \in [1, 10^4]$, $1 \le k \le n$, node values $\in [0, 10^4]$.
- **The Core Invariant: BST Inorder Traversal is Monotonically Sorted:**
  - By the fundamental definition of a BST:
    $$\text{Left Subtree} < \text{Root} < \text{Right Subtree}$$
  - An **Inorder Traversal** (`Left -> Root -> Right`) visits every node in strictly ascending sorted order:
    $$\text{1st visited node} = \text{Smallest Element}$$
    $$\text{2nd visited node} = \text{2nd Smallest Element}$$
    $$\dots$$
    $$k^{\text{th}}\text{ visited node} = k^{\text{th}}\text{ Smallest Element}$$
  - Direct connection to **LeetCode 0098 (Validate BST)**: While problem 98 checks that the inorder sequence strictly ascends, problem 230 takes advantage of this ascending guarantee to pluck the $k^{\text{th}}$ item directly.
- **Why Naive Approaches Fall Short:**
  - Extracting all nodes into a dynamic array and sorting costs $O(N \log N)$ time and $O(N)$ space, completely discarding the inherent ordering already established by the BST structure.
  - Running a full recursive inorder to collect elements in a vector takes $O(N)$ time and $O(N)$ memory, traversing all remaining $N - k$ nodes unnecessarily even when $k$ is tiny (e.g. $k = 1$).
- **The Power of Iterative Inorder with Stack:**
  - By simulating inorder traversal with an explicit stack, we can break out and return immediately as soon as we visit the $k^{\text{th}}$ node.
  - Time is bounded by $O(H + k)$ rather than $O(N)$, visiting only the minimum necessary ancestors and subtrees.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Full Traversal into Array ($O(N)$ time, $O(N)$ space):**
   - Perform recursive DFS, collect all node values into `vector<int>`, and return `vec[k - 1]`.
   - *Drawback:* Wastes memory allocating a full container and unnecessarily visits all $N$ nodes.

2. **Approach 2 — Iterative Inorder Traversal using Stack (Chosen Optimal Solution):**
   - Use an explicit `vector<TreeNode*> stack` and pointer `curr = root`.
   - Loop while `curr != nullptr` or `!stack.empty()`:
     1. **Drill Down Left:** While `curr != nullptr`, push `curr` onto the stack and move `curr = curr->left`. The stack stores ancestors waiting to be visited in bottom-up ascending order.
     2. **Visit Top Node:** Pop the topmost element from the stack (`curr = stack.back(); stack.pop_back();`).
     3. **Count & Early Exit:** Decrement `--k`. If $k == 0$, return `curr->val` immediately.
     4. **Advance to Right Subtree:** Set `curr = curr->right`. In the next iteration, the inner loop will search for the leftmost (minimum) node of this right subtree.
   - *Verdict:* Optimal $O(H + k)$ runtime and $O(H)$ auxiliary space. Halts the very instant the $k^{\text{th}}$ element is uncovered without recursive frame overhead.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(H + k)$
  - Where $H$ is the height of the BST and $k$ is the target rank.
  - Drilling down the left spine to reach the tree's global minimum takes $O(H)$ steps.
  - From that point, the algorithm visits exactly $k$ nodes in sorted order before returning.
  - **Balanced BST:** $H = O(\log N)$, yielding $O(\log N + k)$ time.
  - **Degenerate (Skewed) BST:** $H = O(N)$, yielding $O(N)$ worst-case time.
- **Space Complexity:** $O(H)$
  - Governed entirely by the elements held in the explicit stack.
  - The stack never exceeds the height of the tree ($H$).
  - **Balanced BST:** $O(\log N)$ auxiliary space.
  - **Degenerate BST:** $O(N)$ worst-case space.

---

## 4. Edge Cases & Gotchas

- [x] **$k = 1$ (Minimum Element):**
  - Drills straight down to the leftmost leaf. The very first pop triggers $k = 0$ and returns immediately in $O(H)$ time.
- [x] **$k = N$ (Maximum Element):**
  - Traverses the entire tree, popping the rightmost node at the final step.
- [x] **Left-Skewed Tree:**
  - Pushes all $N$ nodes into the stack before the first pop. Stack space is $O(N)$, but each pop immediately answers each successive rank.
- [x] **Right-Skewed Tree:**
  - Each node has no left child; it is pushed and popped immediately. Stack size never exceeds $1$ ($O(1)$ space).
- [x] **Single-Node Tree ($N = 1, k = 1$):**
  - Pushes root, pops root, decrements $k$ to $0$, and safely returns `root->val`.

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
public:
    int kthSmallest(TreeNode* root, int k) {
        vector<TreeNode*> stack;
        TreeNode* curr = root;

        while (curr || !stack.empty()) {
            // Go as far left as possible.
            while (curr) {
                stack.push_back(curr);
                curr = curr->left;
            }

            // Visit the next node in inorder.
            curr = stack.back();
            stack.pop_back();
            --k;

            if (k == 0) {
                return curr->val;
            }

            // Continue with the right subtree.
            curr = curr->right;
        }

        return -1;
    }
};
```

---

## 6. Visual Walkthrough & Architectural Pattern

### Step-by-Step Execution Trace

Given BST and $k = 3$:

```text
          5
         / \
        3   6
       / \
      2   4
     /
    1
```

#### Step 1: Initial Drill Down Left
- `curr = 5` $\to$ push `5`, `curr = 3`
- `curr = 3` $\to$ push `3`, `curr = 2`
- `curr = 2` $\to$ push `2`, `curr = 1`
- `curr = 1` $\to$ push `1`, `curr = nullptr`
- **Stack State:** `[5, 3, 2, 1]`

#### Step 2: Visiting 1st Smallest
- Pop `1`. Decrement `k`: $k = 3 - 1 = 2$.
- `curr = 1->right` = `nullptr`.
- **Stack State:** `[5, 3, 2]`

#### Step 3: Visiting 2nd Smallest
- Inner while-loop skipped (`curr == nullptr`).
- Pop `2`. Decrement `k`: $k = 2 - 1 = 1$.
- `curr = 2->right` = `nullptr`.
- **Stack State:** `[5, 3]`

#### Step 4: Visiting 3rd Smallest ($k = 0$)
- Inner while-loop skipped (`curr == nullptr`).
- Pop `3`. Decrement `k`: $k = 1 - 1 = 0$.
- $k == 0 \implies$ **Immediate Early Return:** `curr->val = 3`.
- Subtrees rooted at `4`, `5`, and `6` are never traversed!

---

### The Architectural Mental Model

```text
                       BST Inorder Traversal
                                 │
                         Go left as far as
                         possible into stack
                                 ▼
                     ┌───────────────────────┐
                     │   stack.push_back()   │
                     │   curr = curr->left   │
                     └───────────┬───────────┘
                                 │ curr == null
                                 ▼
                     ┌───────────────────────┐
                     │   curr = stack.back() │
                     │   stack.pop_back()    │
                     │   --k                 │
                     └───────────┬───────────┘
                                 │
                     ┌───────────┴───────────┐
            k == 0   ▼                       ▼  k > 0
              Return curr->val        curr = curr->right
             (Early Exit Hit)         (Drill left on right branch)
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** An Inorder Traversal on a BST visits nodes in strictly ascending sorted order. Using an explicit stack permits early termination at step $k$, achieving optimal $O(H + k)$ time and $O(H)$ space.

