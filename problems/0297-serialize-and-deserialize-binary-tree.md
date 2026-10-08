# 0297. Serialize and Deserialize Binary Tree

- **Problem Link:** https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
- **Difficulty:** `Hard`
- **NeetCode Category:** `Trees`
- **LeetCode Topics:** `String` / `Tree` / `Depth-First Search` / `Breadth-First Search` / `Design` / `Binary Tree`
- **Core Pattern:** `Preorder DFS Serialization with Null Sentinel (#) & Stream Deserialization`
- **Last Practiced:** 2026-10-08
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Design an algorithm to serialize a binary tree into a string and deserialize that string back into the original binary tree structure.
  - There is no restriction on how the serialization/deserialization format is formatted, as long as the binary tree can be deserialized to the original tree structure.
  - Constraints: The number of nodes is in $[0, 10^4]$, and node values are in $[-1000, 1000]$.
- **The Ambiguity of Standard Traversals:**
  - In a standard binary tree, a single traversal order (such as preorder or inorder) without null markers cannot uniquely reconstruct the tree structure because multiple different trees can share the identical traversal sequence.
  - Usually, reconstructing a tree requires two traversals (e.g., Preorder + Inorder as in LeetCode 0105), which additionally assumes unique node values.
- **The Core Invariant: Null Sentinel Encoding (`#`):**
  - If we explicitly record empty child pointers as a sentinel symbol (such as `#`), a single **Preorder Traversal** (`Root -> Left -> Right`) becomes a bijective (1-to-1) representation of the tree structure:
    - The first token is always the current subtree's root.
    - If the token is `#`, the node is `nullptr`.
    - Otherwise, create `TreeNode(val)`, and the subsequent tokens recursively build its left subtree followed by its right subtree.
- **Token Separation & Parsing Strategy:**
  - Delimiting tokens with whitespace (`' '`) allows `std::istringstream` to effortlessly parse tokens sequentially without manual index slicing, automatically handling multi-digit integers and negative numbers (e.g. `-1000`).

---

## 2. Approach & Trade-offs

1. **Approach 1 — BFS / Level-Order Traversal with Queue:**
   - Serialize nodes level by level using a queue, appending `#` for null pointers.
   - Deserialization processes tokens using another queue to wire left and right children.
   - *Drawback:* Requires managing queues and explicit child index tracking during deserialization; slightly more verbose than recursive DFS.

2. **Approach 2 — Preorder DFS with Sentinel `#` and `std::istringstream` (Chosen Optimal Solution):**
   - **Serialization (`serialize`):**
     - Perform a standard preorder traversal:
       - If `node == nullptr`, append `"# "` to the output string.
       - Otherwise, append `to_string(node->val) + " "` and recurse on `node->left`, then `node->right`.
   - **Deserialization (`deserialize`):**
     - Wrap the serialized string in `std::istringstream in(data)`.
     - Read the next token: `in >> token`.
     - If `token == "#"` or the stream is empty, return `nullptr`.
     - Otherwise, construct `TreeNode* node = new TreeNode(stoi(token))`.
     - Recurse: `node->left = deserialize(in)` and `node->right = deserialize(in)`.
     - Return `node`.
   - *Verdict:* Optimal $O(N)$ time and $O(N)$ space. Passing `istringstream&` by reference naturally maintains the cursor position across recursive frames, yielding an elegant, symmetric solution.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(N)$
  - **Serialization:** Every node is visited once. For a tree with $N$ nodes, there are exactly $N + 1$ null pointers. Thus, $2N + 1$ tokens are written. Each token conversion takes $O(1)$ amortized time. Total: $O(N)$.
  - **Deserialization:** Each token is extracted from `std::istringstream` and processed once ($2N + 1$ tokens total). String-to-integer conversion (`stoi`) takes $O(1)$ time for bounded values in $[-1000, 1000]$. Total: $O(N)$.
- **Space Complexity:** $O(N)$
  - **Serialized Data:** The serialized string stores $2N + 1$ tokens separated by spaces, taking $O(N)$ auxiliary memory.
  - **Call Stack:** Governed by the height of the tree $H$.
    - Balanced tree: $O(\log N)$ stack frames.
    - Skewed tree: $O(N)$ stack frames in the worst case.
  - Total auxiliary space is strictly $O(N)$.

---

## 4. Edge Cases & Gotchas

- [x] **Empty Tree (`root == nullptr`):** Serializes to `"# "`. Deserialization immediately reads `"#"` and returns `nullptr` safely.
- [x] **Negative Node Values (e.g., `-1`, `-1000`):** `to_string` prepends the minus sign, and `std::istringstream` / `stoi` correctly parses signed integers without custom sign-detection logic.
- [x] **Multi-Digit Numbers:** Using space delimiter prevents ambiguous character sequences (e.g. distinguishing `1` then `2` from `12`).
- [x] **Pass Stream by Reference:** The `istringstream& in` must be passed by reference to ensure all recursive calls consume tokens from the shared stream sequentially without state rollbacks.

---

## 5. Clean Code

```cpp
#include <string>
#include <sstream>

using namespace std;

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 * };
 */
class Codec {
private:
    void serialize(TreeNode* node, string& result) {
        if (!node) {
            result += "# ";
            return;
        }

        result += to_string(node->val);
        result += ' ';
        serialize(node->left, result);
        serialize(node->right, result);
    }

    TreeNode* deserialize(istringstream& in) {
        string token;
        if (!(in >> token) || token == "#") {
            return nullptr;
        }

        TreeNode* node = new TreeNode(stoi(token));
        node->left = deserialize(in);
        node->right = deserialize(in);
        return node;
    }

public:
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        string result;
        serialize(root, result);
        return result;
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        istringstream in(data);
        return deserialize(in);
    }
};

// Your Codec object will be instantiated and called as such:
// Codec ser, deser;
// TreeNode* ans = deser.deserialize(ser.serialize(root));
```

---

## 6. Visual Walkthrough & Architectural Pattern

### Step-by-Step Execution Trace

Consider the binary tree:

```text
       1
      / \
     2   3
        / \
       4   5
```

#### Step 1: Preorder Serialization
Traversing `Root -> Left -> Right`:
- Visit `1`: write `"1 "`
- Visit `2`: write `"2 "`
  - Left child is null: write `"# "`
  - Right child is null: write `"# "`
- Visit `3`: write `"3 "`
  - Visit `4`: write `"4 "`
    - Left null: write `"# "`
    - Right null: write `"# "`
  - Visit `5`: write `"5 "`
    - Left null: write `"# "`
    - Right null: write `"# "`

**Resulting string:** `"1 2 # # 3 4 # # 5 # # "`

#### Step 2: Stream Deserialization
Tokens consumed from `istringstream`:

```text
Token Stream: [ 1 | 2 | # | # | 3 | 4 | # | # | 5 | # | # ]
                │   │   │   │   │   │   │   │   │   │   │
                │   │   │   │   │   │   │   │   │   └───┘── 5->right = null
                │   │   │   │   │   │   │   │   └────────── 5->left  = null
                │   │   │   │   │   │   └───┴────────────── 4->right = null, 4->left = null
                │   │   │   │   │   └────────────────────── 3->left = Node(4)
                │   │   │   │   └────────────────────────── 1->right = Node(3), then 3->right = Node(5)
                │   │   └───┴────────────────────────────── 2->left = null, 2->right = null
                │   └────────────────────────────────────── 1->left = Node(2)
                └────────────────────────────────────────── root = Node(1)
```

The reconstructed tree exactly mirrors the original tree.

---

### The Architectural Mental Model

```text
                  Serialization (Tree -> String)
               ┌───────────────────────────────────┐
               │           Preorder DFS            │
               │  node == null  ──>  append "# "   │
               │  node != null  ──>  val + L + R   │
               └─────────────────┬─────────────────┘
                                 │
                         "1 2 # # 3 # # "
                                 │
                                 ▼
                 Deserialization (String -> Tree)
               ┌───────────────────────────────────┐
               │        istringstream (FIFO)       │
               │  token == "#"  ──>  return null   │
               │  token != "#"  ──>  new Node(val) │
               │                     node->left    │
               │                     node->right   │
               └───────────────────────────────────┘
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** A standard preorder traversal uniquely encodes a binary tree once null pointers are serialized as explicit sentinels (`#`), allowing reconstruction via a single pass over a token stream.

