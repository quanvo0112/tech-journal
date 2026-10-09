# 1046. Last Stone Weight

- **Problem Link:** https://leetcode.com/problems/last-stone-weight/
- **Difficulty:** `Easy`
- **NeetCode Category:** `Heap / Priority Queue`
- **LeetCode Topics:** `Array` / `Heap (Priority Queue)`
- **Core Pattern:** `Max-Heap Simulation / Greedy Pairwise Smashing`
- **Last Practiced:** 2026-10-09
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given an array of integers `stones`, where `stones[i]` represents the weight of the $i^{\text{th}}$ stone.
  - In each step, we choose the **two heaviest stones** with weights $x$ and $y$ where $x \le y$.
  - If $x == y$, both stones are destroyed.
  - If $x \ne y$, stone $x$ is destroyed, and stone $y$ has new weight $y - x$.
  - The game concludes when at most one stone remains. Return the weight of the last remaining stone, or `0` if none are left.
  - Constraints: $1 \le \text{stones.length} \le 30$, $1 \le \text{stones}[i] \le 1000$.
- **The Core Pattern: Dynamic Maximum Extraction:**
  - The problem requires repeatedly querying the maximum and second-maximum elements while dynamically reinserting resultant differences.
  - A **Max-Heap** (`std::priority_queue<int>`) is the ideal data structure for this pattern:
    - Root access `top()` yields the current maximum in $O(1)$ time.
    - Extracting a maximum takes $O(\log N)$ time.
    - Inserting a newly smashed remainder takes $O(\log N)$ time.
- **Why Max-Heap instead of Repeated Array Sorting:**
  - Sorting an array takes $O(N \log N)$. Re-sorting after every smash leads to $O(N^2 \log N)$ runtime.
  - A Max-Heap keeps elements semi-sorted with logarithmic updates, bounding total simulation time to $O(N \log N)$.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Repeated Sorting / Dynamic Vector ($O(N^2 \log N)$):**
   - Sort the array on every collision pass, pop the last two elements, and append their difference.
   - *Drawback:* Redundant sorting operations over a sequence that is modified by only a single element per round.

2. **Approach 2 — Max-Heap Simulation (Chosen Optimal Solution):**
   - Initialize a max-heap in linear $O(N)$ time using the range constructor:
     `priority_queue<int> maxHeap(stones.begin(), stones.end());`
   - While `maxHeap.size() > 1`:
     1. Extract the heaviest stone: $y = \text{maxHeap.top()}$; `maxHeap.pop();`
     2. Extract the second-heaviest stone: $x = \text{maxHeap.top()}$; `maxHeap.pop();`
     3. If $x \ne y$, push the remaining difference $y - x$ back into `maxHeap`.
   - Once at most one stone remains:
     - If `maxHeap.empty()`, return `0` (all stones mutually destroyed).
     - Otherwise, return `maxHeap.top()`.
   - *Verdict:* Optimal $O(N \log N)$ runtime and $O(N)$ space. Each round destroys at least one stone, bounding the loop to at most $N - 1$ iterations.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(N \log N)$
  - **Heap Initialization:** Constructing the heap from iterators uses bottom-up heapification (`std::make_heap`), taking linear $O(N)$ time.
  - **Smash Rounds:** Each round removes two stones and pushes at most one stone back, reducing total stones by at least $1$. Thus, there are at most $N - 1$ rounds.
  - Each round executes at most two pops and one push, each costing $O(\log N)$ time.
  - Total time: $O(N) + (N - 1) \times O(\log N) = O(N \log N)$.
- **Space Complexity:** $O(N)$
  - The `std::priority_queue` stores all $N$ elements initially, requiring $O(N)$ auxiliary memory.

---

## 4. Edge Cases & Gotchas

- [x] **Single Stone ($N = 1$):**
  - The loop condition `maxHeap.size() > 1` is immediately false; returns `maxHeap.top()` safely without any smash iterations.
- [x] **All Stones Destroyed (e.g., `[2, 2]`):**
  - Two equal stones destroy each other, leaving the heap completely empty. The ternary operator `maxHeap.empty() ? 0 : maxHeap.top()` safely returns `0` without attempting an undefined `top()` call on an empty container.
- [x] **Identical Elements in Heap:**
  - Handled seamlessly; priority queues preserve duplicate integers without conflict.
- [x] **Non-Negative Remainder Guarantee:**
  - Because $y$ is extracted before $x$ in a max-heap, $y \ge x$ is guaranteed. Thus, $y - x$ is always strictly positive when $x \ne y$.

---

## 5. Clean Code

```cpp
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    int lastStoneWeight(vector<int>& stones) {
        // Initialize max-heap using range constructor in O(N) time
        priority_queue<int> maxHeap(stones.begin(), stones.end());

        while (maxHeap.size() > 1) {
            int y = maxHeap.top(); // Heaviest stone
            maxHeap.pop();

            int x = maxHeap.top(); // Second heaviest stone
            maxHeap.pop();

            if (x != y) {
                maxHeap.push(y - x);
            }
        }

        return maxHeap.empty() ? 0 : maxHeap.top();
    }
};
```

---

## 6. Visual Walkthrough & Architectural Pattern

### Step-by-Step Execution Trace

Given `stones = [2, 7, 4, 1, 8, 1]`:

#### Step 1: Initial Max-Heap
- `maxHeap` contents: `[8, 7, 4, 1, 2, 1]`

#### Step 2: Simulation Rounds

| Round | Popped $y$ (Max) | Popped $x$ (2nd Max) | Condition | Action | Resulting Heap |
|:---:|:---:|:---:|:---:|:---|:---|
| **1** | `8` | `7` | $8 \ne 7$ | Push $8 - 7 = 1$ | `[4, 2, 1, 1, 1]` |
| **2** | `4` | `2` | $4 \ne 2$ | Push $4 - 2 = 2$ | `[2, 1, 1, 1]` |
| **3** | `2` | `1` | $2 \ne 1$ | Push $2 - 1 = 1$ | `[1, 1, 1]` |
| **4** | `1` | `1` | $1 == 1$ | Both destroyed | `[1]` |

Heap size is now `1`. Loop terminates.
**Return:** `maxHeap.top()` = **1**.

---

### The Architectural Mental Model

```text
                     Initial Stones Array
                              │
                     Heapify into Max-Heap
                              │
                              ▼
                ┌───────────────────────────┐
                │  while maxHeap.size() > 1 │
                └─────────────┬─────────────┘
                              │
                    Pop y (heaviest)
                    Pop x (2nd heaviest)
                              │
                              ▼
                          x == y ?
                         /        \
                   Yes  /          \  No
                       ▼            ▼
                  Both destroyed   maxHeap.push(y - x)
                       \            /
                        ▼          ▼
                     Continue loop iteration
                              │
                              ▼
                  maxHeap.empty() ? 0 : top()
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** Whenever a problem iteratively requires the largest two elements and adds back modified remainders, a Max-Heap provides the optimal $O(N \log N)$ simulation structure.

