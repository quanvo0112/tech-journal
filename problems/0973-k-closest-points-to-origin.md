# 0973. K Closest Points to Origin

- **Problem Link:** https://leetcode.com/problems/k-closest-points-to-origin
- **Difficulty:** Easy
- **NeetCode Category:** Heap / Priority Queue
- **LeetCode Topics:** 2D Geometry, Heaps
- **Core Pattern:** Max-Heap of size k to keep closest points
- **Last Practiced:** 2026-10-12
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly

---

## 1. Pattern Recognition & Triggers

Key phrases in the problem:
- "closest points to origin"
- "keep only k nearest points"
- Comparison of distances

Core intuition:
- We need to find the k points closest to (0,0)
- Instead of sorting all points (O(n log n)), we can use a max-heap of size k
- The heap stores pairs of (distance squared, index) to efficiently keep the k smallest distances
- When the heap exceeds size k, we remove the farthest point

---

## 2. Approach & Trade-offs

### Approach 1 (Brute Force):
Sort all points by distance and take first k → O(n log n)

### Approach 2 (Optimal - Max-Heap):
- Iterate through all points
- Maintain a max-heap of size k containing the k closest points seen so far
- For each point, if heap size < k, push it; else if current point is closer than the farthest in heap, replace
- Final heap contains the k closest points
- Time: O(n log k), Space: O(k)

This is optimal because we never store more than k elements and each insertion/extraction takes O(log k).

---

## 3. Complexity Analysis

- **Time Complexity:** O(n log k) — We process n points, each heap operation costs O(log k)
- **Space Complexity:** O(k) — The heap holds at most k elements

---

## 4. Edge Cases & Gotchas

- Empty input: Return empty list
- Single element: Return that element
- Multiple points with same distance: Any k of them is acceptable
- Large coordinates: Use squared distance to avoid floating-point sqrt

---

## 5. Clean Code

```cpp
class Solution {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int k) {
        // Max-heap: stores {squared distance, index}
        priority_queue<pair<long long, int>> maxHeap;

        for (int i = 0; i < points.size(); ++i) {
            long long x = points[i][0];
            long long y = points[i][1];
            long long dist = x * x + y * y; // Squared distance

            maxHeap.push({dist, i});

            // Keep only k closest points
            if (maxHeap.size() > k) {
                maxHeap.pop();
            }
        }

        vector<vector<int>> ans;
        while (!maxHeap.empty()) {
            ans.push_back(points[maxHeap.top().second]);
            maxHeap.pop();
        }

        return ans;
    }
};
```

---

## 6. Review & Takeaways

* Next Review Date: 
* Key Takeaway: Using a max-heap of size k is an efficient way to find the k closest elements without sorting the entire array. The trick is to compare squared distances to avoid expensive square root calculations.
