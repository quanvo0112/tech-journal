# 295. Find Median from Data Stream

- **Problem Link:** https://leetcode.com/problems/find-median-from-data-stream/
- **Difficulty:** `Hard`
- **NeetCode Category:** `Heap / Priority Queue`
- **LeetCode Topics:** `Sorting` / `Heap (Priority Queue)` / `Design` / `Data Stream`
- **Core Pattern:** `Two Heaps: Max-Heap for Lower Half + Min-Heap for Upper Half`
- **Last Practiced:** 2026-10-15
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Maintain the median of a continuously growing data stream with interleaved `addNum` and `findMedian` calls.
  - Median is defined as the middle value of the sorted order (or the average of the two middle values for even length).
  - `findMedian` must be fast without re-sorting all data after every insertion.
  - Keywords such as "streaming median", "dynamic order statistic", and "insert + median query" point to Two Heaps, not full sorting.
- **Core intuition:**
  - The median splits the sorted sequence into a lower half and an upper half; only the boundary values matter: the maximum of the lower half and the minimum of the upper half.
  - Maintain two invariants:
    - Ordering: every element in `small` <= every element in `large`.
    - Balance: `small.size() == large.size()` or `small.size() == large.size() + 1`.
  - With these invariants, the median is either `small.top()` (odd count) or the average of the two heap tops (even count), giving `O(1)` queries.
- **Why naive approaches fall short:**
  - Keeping a sorted array and inserting each number in order costs `O(n)` shifts per `addNum`.
  - Re-sorting from scratch on every query costs `O(n log n)` per `addNum` — infeasible for up to `5 * 10^4` calls.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Sorted Array with Insertion (Naive):**
   - Store all numbers in a vector kept sorted; insert each new `num` at its sorted position via binary search plus shift, then read the middle element(s).
   - Time `O(n)` per `addNum` due to shifting, `O(1)` per `findMedian`. Auxiliary space `O(1)` besides storage, total space `O(n)`.
   - *Drawback:* Linear insertion dominates for long streams.

2. **Approach 2 — Two Heaps with Push-Then-Rebalance (Chosen Optimal Solution):**
   - Step 1 — Push `num` into `small` (max-heap holding the lower half).
   - Step 2 — Move the largest element of the lower half into `large` (min-heap holding the upper half): `large.push(small.top()); small.pop();`. This restores the ordering invariant regardless of where `num` belongs.
   - Step 3 — Rebalance sizes: if `large.size() > small.size()`, move `large.top()` back to `small`. This restores the balance invariant (`small` holds the extra element when the total count is odd).
   - *Verdict:* Chosen because the unconditional push-then-move sequence needs no explicit comparison branch (`num <= small.top() ? ... : ...`) yet always preserves both invariants. Each `addNum` costs one push plus one pop per heap (`O(log n)`), and `findMedian` is a direct `top()` read in `O(1)`.

---

## 3. Complexity Analysis

- **Time Complexity:** `addNum` $O(\log n)$ — Explanation: each insertion performs a constant number of heap push/pop operations on heaps of total size $n$. `findMedian` $O(1)$ — Explanation: only reads one or two heap tops.
- **Space Complexity:** $O(n)$ — Explanation: the two heaps jointly store all $n$ inserted numbers; no other structure scales with input size.

---

## 4. Edge Cases & Gotchas

- [x] **First insertion:** `large` is empty, so `findMedian` logic must handle `large.top()` being unavailable; the balance rule guarantees `small` holds the single element.
- [x] **Even vs. odd counts:** odd total means `small` has exactly one extra element and the median is `small.top()`; even total means both heaps are equal and the median is the average of the two tops.
- [x] **Duplicate values:** heaps handle equal values naturally; the ordering invariant uses `<=`, so duplicates can sit on either side of the boundary.
- [x] **Negative numbers:** heap comparisons work for the full `[-10^5, 10^5]` range with no special-casing.
- [x] **Integer overflow in averaging:** `(small.top() + large.top())` can overflow `int`; cast to `double` before adding.
- Common pitfalls during implementation:
  - Forgetting the second rebalance step, leaving `large` larger than `small` and breaking the odd-count median rule.
  - Using integer division when averaging, which truncates the `.5` case.

---

## 5. Clean Code

```cpp
#include <queue>
#include <vector>

using namespace std;

class MedianFinder {
private:
    // Max-heap: lower half of the data.
    priority_queue<int> small;

    // Min-heap: upper half of the data.
    priority_queue<int, vector<int>, greater<int>> large;

public:
    MedianFinder() {}

    void addNum(int num) {
        small.push(num);

        // Move the largest element of the lower half to the upper half.
        large.push(small.top());
        small.pop();

        // Keep small equal to large, or exactly one element larger.
        if (large.size() > small.size()) {
            small.push(large.top());
            large.pop();
        }
    }

    double findMedian() {
        if (small.size() > large.size()) {
            return small.top();
        }

        return (static_cast<double>(small.top()) + large.top()) / 2.0;
    }
};
```

---

## 6. Visual Walkthrough & Architectural Pattern

### Step-by-Step Execution Trace

Insert `1`, then `2`, then `3`:

```text
add(1): small=[1] -> move 1 to large -> large=[1], small=[]
        large bigger, move back -> small=[1], large=[]
        median = small.top() = 1.0

add(2): small=[1,2] (top 2) -> move 2 to large -> small=[1], large=[2]
        sizes equal, stop
        median = (1 + 2) / 2.0 = 1.5

add(3): small=[1,3] (top 3) -> move 3 to large -> small=[1], large=[2,3]
        large bigger, move back -> small=[1,2] (top 2), large=[3]
        median = small.top() = 2.0
```

| After adding | `small.top()` | `large.top()` | Median |
|:---:|:---:|:---:|:---:|
| `1` | 1 | — | 1.0 |
| `1, 2` | 1 | 2 | 1.5 |
| `1, 2, 3` | 2 | 3 | 2.0 |

### The Architectural Mental Model

```text
addNum(num) -> push to small (lower half)
         |
         ▼
move small.top() -> large (restore ordering invariant)
         |
         ▼
large bigger than small? -> move large.top() back (restore balance invariant)
         |
         ▼
findMedian -> odd? small.top() : average(small.top(), large.top())
```

- The max-heap exposes the largest of the lower half; the min-heap exposes the smallest of the upper half.
- Those two boundary values are sufficient to answer the median without touching the rest of the data.

* *Next Review Date:*
* *Key Takeaway:* To maintain the median of a changing stream, split the data into two heaps and preserve ordering plus balance; the heap tops alone determine the median in O(1).
