# 621. Task Scheduler

- **Problem Link:** https://leetcode.com/problems/task-scheduler/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Heap / Priority Queue`
- **LeetCode Topics:** `Array` / `Hash Table` / `Greedy` / `Sorting` / `Heap (Priority Queue)` / `Counting`
- **Core Pattern:** `Greedy Counting + Max-Frequency Frame Formula`
- **Last Practiced:** 2026-10-13
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given tasks labeled `'A'` to `'Z'` and a cooldown `n`: two identical tasks must be separated by at least `n` intervals.
  - CPU can execute tasks in any order; idle intervals are allowed when no eligible task exists.
  - Goal: minimum total intervals (tasks + idles) to finish all tasks.
  - Constraints: $1 \le \text{tasks.length} \le 10^4$, $0 \le n \le 100$.
- **Core intuition:**
  - The most frequent task dictates the skeleton of the schedule. All other tasks can only fill the gaps between its occurrences.
  - This is a frequency-bottleneck problem: the answer is determined by `maxFreq` (highest occurrence count) and `countMax` (how many task types tie for that count).
  - Keywords such as "cooldown", "same task must be separated by `n`", and "minimize total time / idle" point to Greedy + Counting rather than explicit simulation.
- **Why naive simulation falls short:**
  - Simulating each CPU interval with a max-heap costs $O(N \log 26)$ with bookkeeping for cooldown queues.
  - Since the alphabet is fixed at 26 letters, a closed-form count of the mandatory frame is sufficient — no heap or per-interval simulation is needed.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Heap Simulation with Cooldown Queue ($O(N \log 26)$):**
   - Push all frequencies into a max-heap; simulate cycle by cycle of length `n + 1`, always running the most frequent available task.
   - Track cooling tasks in a queue with ready-times; reinsert them when the cooldown expires; count idle slots when the heap is empty but tasks remain.
   - *Drawback:* Correct but overkill. Requires heap + queue bookkeeping and explicit idle counting, while the alphabet size makes the closed form trivial.

2. **Approach 2 — Greedy Counting with Frame Formula (Chosen Optimal Solution):**
   - Step 1: Count frequencies in `freq[26]`; track `maxFreq` in the same pass.
   - Step 2: Count `countMax`, the number of task types whose frequency equals `maxFreq`.
   - Step 3: The most frequent tasks force at least `(maxFreq - 1)` full frames of width `(n + 1)`, plus a final partial row of width `countMax`:
     $$\text{intervals} = (maxFreq - 1) \times (n + 1) + countMax$$
   - Step 4: The answer cannot be smaller than the number of tasks themselves, so return:
     $$\max(|\text{tasks}|,\ \text{intervals})$$
   - *Verdict:* Chosen because it skips simulation entirely. Time $O(N)$, auxiliary space $O(1)$, and the formula directly encodes the idle-vs-dense regimes.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(N)$ — Explanation:
  - One pass over $N$ tasks to build `freq` and `maxFreq`, plus one pass over the fixed 26-letter alphabet to compute `countMax`. Total $O(N + 26) = O(N)$.
- **Space Complexity:** $O(1)$ — Explanation:
  - The frequency array always holds exactly 26 integers regardless of input size; only `maxFreq`, `countMax`, and loop scalars are extra.

---

## 4. Edge Cases & Gotchas

- [x] **Zero cooldown ($n = 0$):**
  - Formula gives $(maxFreq - 1) \times 1 + countMax \le N$, so `max(N, intervals)` correctly returns $N$ with no idles.
- [x] **Single dominant task (e.g., `["A","A","A"], n = 2`):**
  - `maxFreq = 3`, `countMax = 1` gives $(3-1) \times 3 + 1 = 7$ (`A idle idle A idle idle A`). The `max()` guard preserves this idle-heavy answer.
- [x] **Multiple tasks tied for max frequency:**
  - The final row needs `countMax` slots, not 1. For `["A","A","A","B","B","B"], n = 2`, `countMax = 2` gives $2 \times 3 + 2 = 8$, not 7.
- [x] **Enough filler tasks to erase idles:**
  - When $N$ exceeds the frame size, all idle slots are filled and the answer collapses to $N$. Example: `["A","A","A","B","B","B","C","C","D","D"], n = 2` returns 10.
- [x] **Single task ($N = 1$):**
  - `maxFreq = 1` gives $(0)(n+1) + 1 = 1$; `max(1, 1) = 1` for any `n`.
- Common pitfalls during implementation:
  - Forgetting the outer `max()` with `tasks.size()` and returning idle-heavy frames even when tasks fill every gap.
  - Counting `countMax` over all 26 slots without filtering `count == maxFreq` (zero-frequency letters must not be counted).
  - Integer-type mismatch in `max(tasks.size(), intervals)` — cast `tasks.size()` to `int`.

---

## 5. Clean Code

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int leastInterval(vector<char>& tasks, int n) {
        vector<int> freq(26, 0);
        int maxFreq = 0;

        for (char task : tasks) {
            int count = ++freq[task - 'A'];
            maxFreq = max(maxFreq, count);
        }

        int countMax = 0;
        for (int count : freq) {
            if (count == maxFreq) {
                ++countMax;
            }
        }

        int intervals = (maxFreq - 1) * (n + 1) + countMax;

        return max(static_cast<int>(tasks.size()), intervals);
    }
};
```

---

## 6. Visual Walkthrough & Architectural Pattern

### Step-by-Step Execution Trace

Given `tasks = ["A","A","A","B","B","B"]`, `n = 2`:

- Frequencies: `A = 3`, `B = 3`, rest `0`.
- `maxFreq = 3`, `countMax = 2` (both `A` and `B`).
- Frame size: $(3 - 1) \times (2 + 1) + 2 = 8$.
- Task count: $6$. Answer: $\max(6, 8) = 8$.

Frame layout built around the most frequent tasks (`#` = forced slot, `.` = fillable/idle):

```text
Row 1:  A  B  .     <- first occurrences of A, B + 1 gap
Row 2:  A  B  .     <- second occurrences of A, B + 1 gap
Row 3:  A  B        <- final row holds every max-frequency task
```

One optimal schedule realizing 8 intervals:

```text
interval: 1  2  3  4  5  6  7  8
task:     A  B  idle  A  B  idle  A  B
```

### The Architectural Mental Model

```text
               Count frequencies (26 slots)
                          |
                   Find maxFreq + countMax
                          |
                          ▼
          Frame = (maxFreq - 1) * (n + 1) + countMax
                          |
                          ▼
              Answer = max(|tasks|, Frame)
```

- Wide frame + few tasks -> idle-heavy regime, frame wins.
- Many filler tasks -> dense regime, `|tasks|` wins and every idle is absorbed.

* *Next Review Date:*
* *Key Takeaway:* When identical items require a cooldown, the maximum frequency determines the schedule skeleton; all remaining items merely fill its gaps, yielding a closed-form answer with no simulation.
