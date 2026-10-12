# 0215. Kth Largest Element in an Array

- **Problem Link:** https://leetcode.com/problems/kth-largest-element-in-an-array
- **Difficulty:** Medium
- **NeetCode Category:** Heap / Priority Queue
- **LeetCode Topics:** Array, Divide and Conquer, Sorting, Heap (Priority Queue), Quickselect
- **Core Pattern:** Quickselect (Hoare's Selection Algorithm) with three-way partition
- **Last Practiced:** 2026-10-12
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly

---

## 1. Pattern Recognition & Triggers

Key phrases:
- "kth largest element"
- "find without sorting entire array"
- "O(n) average time"

Core intuition:
- Convert kth largest to (n - k)th smallest index in sorted order
- Use Quickselect: pick random pivot, three-way partition into < pivot, == pivot, > pivot
- Only recurse into the partition containing the target index
- Random pivot avoids worst-case O(n²) on sorted/nearly-sorted arrays

---

## 2. Approach & Trade-offs

### Approach 1 (Sorting):
Sort entire array → O(n log n) time, O(1) space

### Approach 2 (Min-Heap of size k):
Maintain min-heap of k largest elements → O(n log k) time, O(k) space

### Approach 3 (Optimal - Quickselect):
- Convert kth largest to target index = n - k
- Pick random pivot, three-way partition (Dutch National Flag)
- If target in left partition (< pivot): recurse left
- If target in right partition (> pivot): recurse right
- If target in middle partition (== pivot): return pivot
- Average O(n), worst O(n²), O(1) space

Quickselect is optimal for this problem because we don't need the full ordering, only the element at a specific rank.

---

## 3. Complexity Analysis

- **Time Complexity:** O(n) average; O(n²) worst case (mitigated by random pivot)
- **Space Complexity:** O(1) auxiliary (iterative implementation avoids recursion stack)

---

## 4. Edge Cases & Gotchas

- Empty array: not possible per constraints (1 ≤ k ≤ n)
- All identical elements: three-way partition handles efficiently
- Single element array: returns that element
- k = 1: returns maximum element
- k = n: returns minimum element
- Large values: use long long if computing indices, but int is fine for n ≤ 10⁵

---

## 5. Clean Code

```cpp
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        int n = nums.size();
        int target = n - k;  // Convert to (n-k)th smallest (0-indexed)

        int left = 0;
        int right = n - 1;

        mt19937 rng(random_device{}());

        while (left <= right) {
            uniform_int_distribution<int> dist(left, right);
            int pivot = nums[dist(rng)];

            // Three-way partition (Dutch National Flag):
            // [left, lt - 1] < pivot
            // [lt, gt]       == pivot
            // [gt + 1, right] > pivot
            int lt = left;
            int i = left;
            int gt = right;

            while (i <= gt) {
                if (nums[i] < pivot) {
                    swap(nums[lt], nums[i]);
                    ++lt;
                    ++i;
                } else if (nums[i] > pivot) {
                    swap(nums[i], nums[gt]);
                    --gt;
                } else {
                    ++i;
                }
            }

            if (target < lt) {
                right = lt - 1;        // Target in left partition
            } else if (target > gt) {
                left = gt + 1;         // Target in right partition
            } else {
                return nums[target];   // Target in middle partition
            }
        }

        return -1;  // Should never reach here
    }
};
```

---

## 6. Review & Takeaways

* Next Review Date: 
* Key Takeaway: Quickselect with random pivot and three-way partition achieves O(n) average time and O(1) space for selection problems. The three-way partition is crucial for handling duplicates efficiently. For kth largest, simply convert to (n - k)th smallest index.