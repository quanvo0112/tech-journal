# 0235. Lowest Common Ancestor of a Binary Search Tree

- **Problem Link:** https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Trees`
- **LeetCode Topics:** `Tree` / `Depth-First Search` / `Binary Search Tree` / `Binary Tree`
- **Core Pattern:** `Iterative BST Search / Split-Point Identification via Ordering Invariant`
- **Last Practiced:** 2026-10-05
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given a **Binary Search Tree (BST)** and two nodes `p` and `q`, find their **Lowest Common Ancestor (LCA)**.
  - The Lowest Common Ancestor is defined between two nodes `p` and `q` as the lowest node in $T$ that has both `p` and `q` as descendants (where a node can be a descendant of itself).
  - Constraints: All node values are unique; $p \neq q$; both $p$ and $q$ are guaranteed to exist in the BST.
- **The BST Ordering Invariant:**
  - For any node $u$ in a BST:
    $$\forall v \in u.\text{left}: v.\text{val} < u.\text{val} \quad \text{and} \quad \forall w \in u.\text{right}: w.\text{val} > u.\text{val}$$
  - Unlike general binary trees where finding LCA requires an exhaustive post-order DFS exploring multiple subtrees, a BST provides deterministic directionality in $O(1)$ comparisons:
    1. **Both nodes in left subtree:** If $p.\text{val} < \text{curr}.\text{val}$ and $q.\text{val} < \text{curr}.\text{val}$, the LCA *must* reside strictly in the left subtree.
    2. **Both nodes in right subtree:** If $p.\text{val} > \text{curr}.\text{val}$ and $q.\text{val} > \text{curr}.\text{val}$, the LCA *must* reside strictly in the right subtree.
    3. **Split Point (LCA Found):** If one node is smaller and the other is larger, or if $\text{curr}$ equals either $p$ or $q$, the paths to $p$ and $q$ diverge at $\text{curr}$. Hence, $\text{curr}$ is the Lowest Common Ancestor.

---

## 2. Approach & Trade-offs

1. **Approach 1 — General Binary Tree DFS ($O(N)$ time, $O(h)$ space):**
   - Run post-order DFS searching for $p$ and $q$ without utilizing the BST ordering property.
   - *Drawback:* Explores unrelated branches unnecessarily, degrading runtime to $O(N)$ instead of $O(h)$.

2. **Approach 2 — Recursive BST Traversal ($O(h)$ time, $O(h)$ space):**
   - Recurse down the tree by comparing values:
     ```cpp
     if (p->val < root->val && q->val < root->val) return lowestCommonAncestor(root->left, p, q);
     if (p->val > root->val && q->val > root->val) return lowestCommonAncestor(root->right, p, q);
     return root;
     ```
   - *Drawback:* Simple tail recursion, but consumes $O(h)$ call stack space.

3. **Approach 3 — Iterative BST Traversal (Chosen Optimal Solution):**
   - Replace recursion with a single `while (root)` loop.
   - Update `root = root->left` or `root = root->right` until the split point is reached.
   - Return `root` immediately at the split.
   - *Verdict:* Optimal $O(h)$ time and strictly $O(1)$ auxiliary space. Zero stack allocation or heap overhead.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(h)$
  - Where $h$ is the height of the BST.
  - At each step of the `while` loop, we make constant-time $O(1)$ comparisons and advance down exactly one level of the tree.
  - **Balanced BST:** $h = O(\log N)$.
  - **Degenerate / Skewed BST:** $h = O(N)$.
- **Space Complexity:** $O(1)$
  - State is tracked purely using pointer reassignment. Zero call stack frames or auxiliary heap data structures are created.

---

## 4. Edge Cases & Gotchas

- [x] **One Node is Ancestor of the Other (Self-Ancestor):** E.g., `p = 2` and `q = 4` with `root = 2`. The conditions `p < root && q < root` and `p > root && q > root` both evaluate to `false` because `p == root`. The loop cleanly falls through to the `else` block and returns `root` (`2`), adhering to the invariant that a node is an ancestor of itself.
- [x] **Nodes on Opposite Sides of Root:** E.g., $p$ is in left subtree and $q$ is in right subtree. The split occurs at `root` immediately, terminating in $O(1)$ time.
- [x] **No Ordering Guarantee on $p$ vs. $q$:** $p$ is not guaranteed to be less than $q$. Testing `p->val < root->val && q->val < root->val` handles any arbitrary order between $p$ and $q$ symmetrically without requiring sorting `min(p, q)` beforehand.

---

## 5. Clean Code (Optimal Solution: Iterative BST Traversal)

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root,
                                   TreeNode* p,
                                   TreeNode* q) {
        while (root) {
            if (p->val < root->val && q->val < root->val) {
                // Both nodes are in the left subtree
                root = root->left;
            }
            else if (p->val > root->val && q->val > root->val) {
                // Both nodes are in the right subtree
                root = root->right;
            }
            else {
                // The split point: paths to p and q diverge, or root is p/q
                return root;
            }
        }

        return nullptr;
    }
};
```

---

## 6. Review & Takeaways

### Visual Split-Point Dry-Run

```text
Given BST:
        6
       / \
      2   8
     / \ / \
    0  4 7  9
      / \
     3   5

Case 1: p = 2, q = 8
- Start at Root 6:
  - 2 < 6 (p is on left)
  - 8 > 6 (q is on right)
  - Split occurs at 6!
  -> Return 6 (LCA)

Case 2: p = 2, q = 4
- Start at Root 6:
  - 2 < 6 and 4 < 6 -> Both in left subtree -> root = root->left (Node 2)
- At Node 2:
  - p->val == 2 (matches root)
  - Neither condition triggers -> Split block reached!
  -> Return 2 (LCA)

Case 3: p = 3, q = 5
- Start at Root 6: Both < 6 -> Go left to 2
- At Node 2:      Both > 2 -> Go right to 4
- At Node 4:      3 < 4 and 5 > 4 -> Divergence split!
  -> Return 4 (LCA)
```

---

### The Architectural Mental Model

```text
               Lowest Common Ancestor of a BST
                               ↓
                        while (root != null)
                               ↓
             p->val < root->val && q->val < root->val ?
                           /        \
                     YES  /          \  NO
                         ↓            ↓
                  root = root->left   p->val > root->val && q->val > root->val ?
                                                    /          \
                                              YES  /            \  NO
                                                  ↓              ↓
                                           root = root->right   return root
                                                                (Split Point / LCA)
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** In a Binary Search Tree, the Lowest Common Ancestor is simply the first node along the downward search path where $p$ and $q$ diverge or match—yielding an optimal $\mathcal{O}(h)$ time and $\mathcal{O}(1)$ space iterative walk.
