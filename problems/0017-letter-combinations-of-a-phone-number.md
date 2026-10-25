# 0017. Letter Combinations of a Phone Number

- **Problem Link:** https://leetcode.com/problems/letter-combinations-of-a-phone-number/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Backtracking`
- **LeetCode Topics:** `Hash Table` / `String` / `Backtracking`
- **Core Pattern:** `Position-Indexed Backtracking / Cartesian Product Enumeration`
- **Last Practiced:** 2026-10-25
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Given a string of digits from `2` to `9`, return **all possible letter combinations** the number could represent.
  - The mapping is fixed by the classic phone keypad; the answer order is arbitrary.
  - Return type is `vector<string>`, and the constraint is $0 \le \text{digits.length} \le 4$.
  - Digits `0` and `1` never appear — there are no "empty" choice sets to handle.
- **Core intuition:**
  - The answer is a **Cartesian product**: one letter is chosen from the set of the first digit, one from the set of the second, and so on. If digit $d$ has $|S_d|$ letters, the result contains $\prod_d |S_d| \le 4^n$ strings.
  - **Each recursion level is an input position, not a value choice.** `backtrack(index)` owns exactly one digit (`digits[index]`), enumerates its 2–4 letters, and delegates the next position to `index + 1`. This is structurally different from Subsets or Permutations:
    - No `start` parameter — there is no "choose elements in increasing index order" constraint to encode.
    - No `used[]` array — positions are consumed strictly left to right, so a letter can never be picked twice for the same position.
    - No duplicate-skipping logic — letters within one key are distinct, and combinations differing at any position are genuinely different outputs.
  - **The recursion invariant:** at depth `index`, `path` contains exactly one chosen letter for each of the digits `digits[0..index-1]`. Reaching `index == n` means every position has been assigned, so `path` is a complete combination.
  - **Choose -> Explore -> Undo:** `path.push_back(c)` -> recurse -> `path.pop_back()` keeps a single mutable buffer alive across the whole tree instead of allocating a fresh string per branch.
  - **Lookup is $O(1)$:** the keypad table is indexed directly by `digits[index] - '0'`, replacing any conditional chain with a table read.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Iterative Cartesian Product expansion ($O(n \cdot 4^n)$ time, $O(n \cdot 4^n)$ space):**
   - Start with `res = {""}`. For each digit, build a new vector: for every string already in `res`, append each letter of that digit.
   - *Drawback:* Reallocates and copies the entire partial result at every digit. It also needs a special case for empty input, because `{""}` would otherwise leak a spurious empty string into the answer. The intermediate vectors double as both state and output, which muddies the logic.
2. **Approach 2 — BFS queue of prefixes ($O(n \cdot 4^n)$ time, $O(4^n)$ space):**
   - Push `""` into a queue; pop a prefix per digit layer, append each letter, and push the extended prefixes back.
   - *Drawback:* Same copying cost as Approach 1, plus queue bookkeeping to know when a layer ends. It is essentially the iterative expansion wearing a different data structure.
3. **Approach 3 — Index-driven Backtracking (Chosen Optimal Solution):**
   - `keypad` is a `const vector<string>` indexed by digit value.
   - `backtrack(digits, index)`:
     - If `index == digits.size()`: `res.push_back(path)` and return.
     - Fetch `letters = keypad[digits[index] - '0']`.
     - For each `c` in `letters`: `path.push_back(c)`, `backtrack(digits, index + 1)`, `path.pop_back()`.
   - Public method clears `res` / `path`, short-circuits on empty `digits`, then starts at `index = 0`.
   - *Verdict:* Optimal and idiomatic. Every emitted node is a complete valid combination, mutations are $O(1)$, auxiliary memory is only the shared buffer plus the recursion stack, and the shape generalizes directly to any per-position choice-set problem.

---

## 3. Complexity Analysis

- **Time Complexity:** $O(n \cdot 4^n)$
  - Each of the $n$ digits branches into at most 4 choices, so the recursion tree has $O(4^n)$ leaves — exactly the number of output combinations in the worst case.
  - Each leaf copies a string of length $n$ into `res`, giving the output-sensitive $O(n)$ factor.
  - Work at internal nodes is proportional to the number of nodes, which is dominated by the $O(n \cdot 4^n)$ leaf-copy cost.
- **Space Complexity:** $O(n)$ auxiliary
  - `path` holds exactly one character per processed digit, so at most $n$.
  - The recursion stack depth is exactly $n + 1$ frames.
  - `res` stores the $O(4^n)$ output strings (each length $n$) and is not counted as auxiliary space.

---

## 4. Edge Cases & Gotchas

- [x] **Empty input (`digits = ""`):** must return `[]`, not `[""]`. The early `if (digits.empty()) return res;` is load-bearing — without it, `backtrack` would immediately hit `index == size()` and push an empty string.
- [x] **Single digit (`"2"`):** returns `["a","b","c"]`; the loop at depth 0 emits each letter and every child depth-1 call saves immediately.
- [x] **Digits `7` and `9`:** carry **4** letters (`"pqrs"`, `"wxyz"`), so the worst-case branching factor is 4, not 3.
- [x] **Digits `0` / `1`:** excluded by constraints; the table maps them to `""`, which would simply produce no branches rather than crash. The indexing `digits[index] - '0'` stays in `[2, 9]`.
- [x] **Leaf length invariant:** at `index == n`, `path.size() == n` automatically; no separate length check is needed.
- [x] **State restoration:** omitting `path.pop_back()` leaves a stale character and every subsequent sibling combination gets an extra leading/garbled character.
- [x] **No `used[]` needed:** positions advance monotonically, and letters in a key are distinct, so duplication is structurally impossible.
- [x] **Repeated calls on one object:** `res.clear()` and `path.clear()` in the public method prevent stale results from a previous invocation.
- **Common pitfall:** using a `start` index copied from the Subsets template. Here the level is the *input position*, so the loop must enumerate the current digit's letters, not indices of `digits`.

---

## 5. Clean Code (Optimal Solution: Backtracking)

```cpp
class Solution {
private:
    const vector<string> keypad = {
        "",     "",     // 0, 1
        "abc",  "def",  // 2, 3
        "ghi",  "jkl",  // 4, 5
        "mno",  "pqrs", // 6, 7
        "tuv",  "wxyz"  // 8, 9
    };

    vector<string> res;
    string path;

    void backtrack(const string& digits, int index) {
        if (index == digits.size()) {
            res.push_back(path);
            return;
        }

        const string& letters = keypad[digits[index] - '0'];

        for (char c : letters) {
            // Choose
            path.push_back(c);

            // Explore the next digit
            backtrack(digits, index + 1);

            // Backtrack
            path.pop_back();
        }
    }

public:
    vector<string> letterCombinations(string digits) {
        res.clear();
        path.clear();

        if (digits.empty()) {
            return res;
        }

        backtrack(digits, 0);
        return res;
    }
};
```

---

## 6. Review & Takeaways

### Visual Recursion Tree (`digits = "23"`)

```text
backtrack(0)                       digit '2' -> "abc"
├── 'a' -> backtrack(1)            digit '3' -> "def"
│   ├── 'd' -> backtrack(2) == n  [SAVE "ad"]
│   ├── 'e' -> backtrack(2) == n  [SAVE "ae"]
│   └── 'f' -> backtrack(2) == n  [SAVE "af"]
├── 'b' -> backtrack(1)
│   ├── 'd' -> backtrack(2) == n  [SAVE "bd"]
│   ├── 'e' -> backtrack(2) == n  [SAVE "be"]
│   └── 'f' -> backtrack(2) == n  [SAVE "bf"]
└── 'c' -> backtrack(1)
    ├── 'd' -> backtrack(2) == n  [SAVE "cd"]
    ├── 'e' -> backtrack(2) == n  [SAVE "ce"]
    └── 'f' -> backtrack(2) == n  [SAVE "cf"]

Final Result: ["ad","ae","af","bd","be","bf","cd","ce","cf"]  (3 x 3 = 9)
```

---

### The Architectural Mental Model

```text
                 Letter Combinations (digits)
                              ↓
                  Empty input? -> return []
                              ↓
                  backtrack(index = 0)
                              ↓
                  ┌───────────┴───────────┐
                  ↓                       ↓
          index == n ?              letters = keypad[digit]
          YES -> save path                 ↓
                                    for c in letters
                                           ↓
                                   push c -> recurse(index + 1)
                                           ↓
                                      pop_back()
```

* **Next Review Date:**
* **Key Takeaway:** Treat each input position as one recursion level: look up that digit's 2–4 letters, push one, recurse on the next position, and only save when `index == digits.size()` — no `used[]` array or `start` bound is needed, but empty input must short-circuit to `[]`.
