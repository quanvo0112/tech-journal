# 0703. Kth Largest Element in a Stream

- **Problem Link:** https://leetcode.com/problems/kth-largest-element-in-a-stream/
- **Difficulty:** `Easy`
- **NeetCode Category:** `Heap / Priority Queue`
- **LeetCode Topics:** `Tree` / `Design` / `Binary Search Tree` / `Heap (Priority Queue)` / `Binary Tree` / `Data Stream`
- **Core Pattern:** `Min-Heap of Fixed Capacity k / Dynamic Bounded Stream Top-K Tracking`
- **Last Practiced:** 2026-10-09
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Design a class to find the $k^{\text{th}}$ largest element in a continuous data stream.
  - Initialized with integer $k$ and an initial integer array `nums`.
  - The `add(int val)` method appends `val` to the stream and returns the element representing the $k^{\text{th}}$ largest element in the stream.
  - Constraints: $1 \le k \le 10^4$, $0 \le \text{nums.length} \le 10^4$, $-10^4 \le \text{nums}[i], \text{val} \le 10^4$, at most $10^4$ calls to `add`.
- **The Core Invariant: The Smallest of the Top-$k$ is the $k^{\text{th}}$ Largest:**
  - We do not need to store or sort all $N$ stream elements.
  - If we maintain a pool of strictly the **$k$ largest elements** observed so far, the **minimum element** within this pool is by definition the **$k^{\text{th}}$ largest overall**.
  - A **Min-Heap** (`priority_queue<int, vector<int>, greater<int>>`) naturally exposes its smallest stored value at `top()` in $O(1)$ time.
- **Why Min-Heap instead of Max-Heap?**
  - A Max-Heap stores the largest elements at the root. To uncover the $k^{\text{th}}$ largest element, one would have to extract $k - 1$ elements, inspect the $k^{\text{th}}$, and push them all back—incurring $O(k \log N)$ per query.
  - A Min-Heap restricted to size $k$ maintains the $k^{\text{th}}$ largest right at `minHeap.top()` in $O(1)$ read time, and insertions cost only $O(\log k)$.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Full Array Re-Sorting on Each Addition ($O(M \cdot N \log N)$):**
   - Append `val` to a dynamic array, sort descending, and return `arr[k - 1]`.
   - *Drawback:* Completely impractical for real-time streams with up to $10^4$ operations.

2. **Approach 2 — Sorted Vector with Binary Search Insertion ($O(M \cdot N)$):**
   - Maintain a sorted array of size $k$. Use `std::lower_bound` to insert each incoming element and drop elements beyond index $k$.
   - *Drawback:* Shifting elements in an array takes $O(k)$ linear time per addition.

3. **Approach 3 — Min-Heap of Fixed Capacity $k$ (Chosen Optimal Solution):**
   - Maintain a min-heap with a maximum capacity of $k$.
   - **Constructor:** Reuses `add(num)` for each element in initial `nums`, ensuring a single unified code path.
   - **`add(val)` Method:**
     1. Push `val` into `minHeap`.
     2. If `minHeap.size() > k`, evict the root via `minHeap.pop()`.
     3. Return `minHeap.top()`.
   - *Verdict:* Optimal $O(\log k)$ update time, $O(1)$ retrieval, and strict $O(k)$ memory footprint regardless of stream scale.

---

## 3. Complexity Analysis

- **Time Complexity:**
  - **Constructor:** $O(N \log k)$, where $N = \text{nums.length}$. Each of the $N$ initial insertions into a heap bounded at size $k + 1$ costs $O(\log k)$ time.
  - **`add(val)`:** $O(\log k)$. Inserting an element into a heap of size $k$ takes $O(\log k)$ push time, and an optional pop takes $O(\log k)$. Inspecting `minHeap.top()` takes $O(1)$.
- **Space Complexity:** $O(k)$
  - The heap retains at most $k$ elements (momentarily $k + 1$ before `pop()`).
  - Auxiliary memory is independent of total stream length or initial array size $N$.

---

## 4. Edge Cases & Gotchas

- [x] **Initial `nums.length < k`:** The problem guarantees at least $k$ elements exist whenever the $k^{\text{th}}$ largest is retrieved. When initial array size is smaller than $k$, elements simply accumulate in the heap without popping until size exceeds $k$.
- [x] **Duplicate Values:** Handled automatically; priority queues correctly manage identical values according to standard ordering without perturbing rank counts.
- [x] **Negative Numbers:** Priority queue comparisons natively support signed integers (e.g. $[-10^4, 10^4]$).
- [x] **$k = 1$:** The heap maintains a single element, which acts as a running maximum tracker with $O(1)$ operations.

---

## 5. Clean Code

```cpp
#include <queue>
#include <vector>

using namespace std;

class KthLargest {
private:
    int k;
    priority_queue<int, vector<int>, greater<int>> minHeap;

public:
    KthLargest(int k, vector<int>& nums) : k(k) {
        for (int num : nums) {
            add(num);
        }
    }

    int add(int val) {
        minHeap.push(val);

        if (minHeap.size() > k) {
            minHeap.pop();
        }

        return minHeap.top();
    }
};

/**
 * Your KthLargest object will be instantiated and called as such:
 * KthLargest* obj = new KthLargest(k, nums);
 * int param_1 = obj->add(val);
 */
```

---

## 6. Visual Walkthrough & Architectural Pattern

### Step-by-Step Execution Trace

Given $k = 3$, initial array `nums = [4, 5, 8, 2]`:

#### Step 1: Initial Construction
- `add(4)` $\to$ heap: `[4]` (size 1)
- `add(5)` $\to$ heap: `[4, 5]` (size 2)
- `add(8)` $\to$ heap: `[4, 5, 8]` (size 3)
- `add(2)` $\to$ heap: `[2, 4, 5, 8]` (size 4 $> k$) $\to$ pop `2` $\to$ heap: `[4, 5, 8]`
- **State after constructor:** `minHeap = [4, 5, 8]`, `top = 4` ($3^{\text{rd}}$ largest).

#### Step 2: Stream Calls
1. **Call `add(10)`:**
   - Push `10` $\to$ `[4, 5, 8, 10]`
   - Size $> 3 \implies$ pop minimum (`4`) $\to$ `[5, 8, 10]`
   - Return `minHeap.top()` = **5**.
2. **Call `add(3)`:**
   - Push `3` $\to$ `[3, 5, 8, 10]`
   - Size $> 3 \implies$ pop minimum (`3`) $\to$ `[5, 8, 10]`
   - Return `minHeap.top()` = **5** (3 was too small to enter the top-3).
3. **Call `add(9)`:**
   - Push `9` $\to$ `[5, 8, 9, 10]`
   - Size $> 3 \implies$ pop minimum (`5`) $\to$ `[8, 9, 10]`
   - Return `minHeap.top()` = **8**.

---

### The Architectural Mental Model

```text
                  Kth Largest Element in a Stream
                                 │
                         Min-Heap of Size k
                                 │
                         Incoming val added
                                 │
                                 ▼
                     minHeap.push(val)
                                 │
                                 ▼
                    minHeap.size() > k ?
                           /    \
                     Yes  /      \  No
                         ▼        ▼
                   minHeap.pop()  keep heap
                         \        /
                          ▼      ▼
                      minHeap.top()
                  (= kth largest element)
```

* **Next Review Date:** As needed / TBD
* **Key Takeaway:** To dynamically track the $k^{\text{th}}$ largest element of an unbounded stream, bound a Min-Heap to capacity $k$; the heap root always holds the minimum of the top-$k$ elements, delivering $O(\log k)$ updates and $O(1)$ lookups.

