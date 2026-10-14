# 355. Design Twitter

- **Problem Link:** https://leetcode.com/problems/design-twitter/
- **Difficulty:** `Medium`
- **NeetCode Category:** `Heap / Priority Queue`
- **LeetCode Topics:** `Hash Table` / `Linked List` / `Design` / `Heap (Priority Queue)`
- **Core Pattern:** `K-way Merge with Max-Heap over Per-User Tweet Streams`
- **Last Practiced:** 2026-10-14
- **Proficiency Level:**
  - [x] Level 1: Solved smoothly (< 20 mins, optimal)
  - [ ] Level 2: Struggled / Non-optimal / Edge-case bugs
  - [ ] Level 3: Needed editorial or hints

---

## 1. Pattern Recognition & Triggers

- **Clues in prompt:**
  - Design `postTweet`, `getNewsFeed`, `follow`, `unfollow` with recency ordering.
  - `getNewsFeed(userId)` must return up to 10 most recent tweets posted by the user and their followees, ordered newest first.
  - Each user has an independent tweet history; the feed is a merge of several recency-ordered streams.
  - Keywords such as "10 most recent across many sorted lists" and "merge without scanning everything" point to K-way Merge, not global sorting.
- **Core intuition:**
  - Each user's tweets are naturally sorted by posting time if stamped with a monotonically increasing global `timestamp`.
  - The feed is therefore a merge of `F + 1` descending-time streams (self + up to `F` followees).
  - A max-heap seeded with one entry per stream (the newest tweet of each author) always yields the globally newest remaining tweet; after popping it, push the next older tweet from the same author.
- **Why naive approaches fall short:**
  - Collecting every tweet from every followee and sorting costs `O(T log T)` per query, where `T` is total visible tweets — far too slow when histories are long.
  - Relying on `tweetId` for ordering is incorrect because IDs do not necessarily reflect posting order; an explicit global timestamp is required.

---

## 2. Approach & Trade-offs

1. **Approach 1 — Collect All and Sort (Naive):**
   - On `getNewsFeed`, gather all `(timestamp, tweetId)` pairs from self plus all followees into one vector, sort descending by timestamp, take the first 10.
   - Time `O(T log T)` per feed query, auxiliary space `O(T)`. Simple but wasteful because only 10 results are ever needed.
   - *Drawback:* Scans and sorts entire histories on every call.

2. **Approach 2 — Hash Map + Hash Set + Max-Heap K-way Merge (Chosen Optimal Solution):**
   - Step 1 — Storage: `tweets` maps `userId -> vector<pair<timestamp, tweetId>>` in posting order (append-only). `following` maps `followerId -> unordered_set<followeeId>` for average `O(1)` follow/unfollow. A global `timestamp` counter is incremented on every `postTweet`.
   - Step 2 — Seed: take the newest tweet (last vector element) of the requester and of each followee, and build a max-heap of entries `{timestamp, authorId, index}`.
   - Step 3 — Merge: pop the heap top (globally newest remaining tweet), append its `tweetId` to the feed, then push the next older tweet (`index - 1`) from the same author if one exists.
   - Step 4 — Stop after 10 tweets or when the heap is empty.
   - *Verdict:* Chosen because it touches only what the feed needs: `O(F)` seeding plus at most 10 heap pops. This is the same K-way Merge technique as Merge K Sorted Lists, traversed newest-to-oldest.

---

## 3. Complexity Analysis

- **Time Complexity:** `postTweet` $O(1)$ amortized (vector append); `follow` / `unfollow` $O(1)$ average (hash set insert/erase); `getNewsFeed` $O(F + 10 \log(F+1))$ — Explanation: seeding the heap from at most $F+1$ authors costs $O(F)$, then at most 10 pop-plus-push rounds each cost $O(\log(F+1))$, where $F$ is the number of followees.
- **Space Complexity:** $O(T + R)$ total, $O(F + 1)$ auxiliary per feed query — Explanation: $T$ tweet pairs and $R$ follow edges persist across calls; each `getNewsFeed` holds at most one heap entry per visible author plus the up-to-10 output.

---

## 4. Edge Cases & Gotchas

- [x] **User follows themselves:** guard `followerId != followeeId` in `follow`/`unfollow`, and skip `followeeId == userId` when seeding to avoid pushing the same stream twice.
- [x] **Author with no tweets:** `tweets.find(authorId)` may miss or hold an empty vector — skip such authors when seeding instead of indexing blindly.
- [x] **Fewer than 10 visible tweets:** return whatever the heap yields; the loop naturally ends when the heap drains.
- [x] **Unfollow non-followed user:** look up `following` first; erasing a missing key must be a no-op, not an accidental map insertion.
- [x] **Ordering by `tweetId`:** never use `tweetId` as a recency proxy; only the global `timestamp` reflects true posting order.
- Common pitfalls during implementation:
  - Using `tweets[authorId]` inside `getNewsFeed` for reads, which inserts empty entries for tweet-less users — prefer `find` / `at`.
  - Forgetting to include the requester's own tweets in the seed set.
  - Missing `#include <tuple>` / structured-binding support for `tuple<int, int, int>` heap entries.

---

## 5. Clean Code

```cpp
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>

using namespace std;

class Twitter {
private:
    // {timestamp, tweetId}, appended in posting order per user.
    unordered_map<int, vector<pair<int, int>>> tweets;

    // followerId -> set of followeeIds.
    unordered_map<int, unordered_set<int>> following;

    int timestamp = 0;

public:
    Twitter() {}

    void postTweet(int userId, int tweetId) {
        tweets[userId].push_back({++timestamp, tweetId});
    }

    vector<int> getNewsFeed(int userId) {
        // Each entry: {timestamp, authorId, indexInTweets}.
        using Entry = tuple<int, int, int>;

        vector<Entry> initial;

        auto addLatestTweet = [&](int authorId) {
            auto it = tweets.find(authorId);
            if (it == tweets.end() || it->second.empty()) {
                return;
            }
            int index = static_cast<int>(it->second.size()) - 1;
            initial.emplace_back(it->second[index].first, authorId, index);
        };

        // Include the user's own tweets.
        addLatestTweet(userId);

        // Include tweets from followed users.
        auto it = following.find(userId);
        if (it != following.end()) {
            for (int followeeId : it->second) {
                if (followeeId != userId) {
                    addLatestTweet(followeeId);
                }
            }
        }

        // Build a max-heap from the newest tweet of each user.
        priority_queue<Entry> maxHeap(initial.begin(), initial.end());

        vector<int> feed;
        feed.reserve(10);

        while (!maxHeap.empty() && feed.size() < 10) {
            auto [time, authorId, index] = maxHeap.top();
            maxHeap.pop();

            const auto& userTweets = tweets.at(authorId);
            feed.push_back(userTweets[index].second);

            // Add the next older tweet from the same author.
            if (index > 0) {
                int previousIndex = index - 1;
                maxHeap.emplace(
                    userTweets[previousIndex].first,
                    authorId,
                    previousIndex
                );
            }
        }

        return feed;
    }

    void follow(int followerId, int followeeId) {
        if (followerId != followeeId) {
            following[followerId].insert(followeeId);
        }
    }

    void unfollow(int followerId, int followeeId) {
        if (followerId != followeeId) {
            auto it = following.find(followerId);
            if (it != following.end()) {
                it->second.erase(followeeId);
            }
        }
    }
};
```

---

## 6. Visual Walkthrough & Architectural Pattern

### Step-by-Step Execution Trace

User 1 follows user 2. Posts in global time order: `A` (user 1, t=1), `B` (user 2, t=2), `C` (user 1, t=3), `D` (user 2, t=4).

Per-user streams (oldest -> newest): user 1 = `[A(1), C(3)]`, user 2 = `[B(2), D(4)]`.

```text
Seed heap with newest of each stream: {C(3), D(4)}
Heap (max first): [D(4), C(3)]

Pop D(4) -> feed [D]; push B(2) from user 2 -> heap [C(3), B(2)]
Pop C(3) -> feed [D, C]; push A(1) from user 1 -> heap [B(2), A(1)]
Pop B(2) -> feed [D, C, B]; no older tweet from that entry chain left behind it
Pop A(1) -> feed [D, C, B, A]; heap empty, stop
```

Only 4 heap pops are needed; the full histories are never concatenated or sorted.

### The Architectural Mental Model

```text
postTweet -> append {++timestamp, tweetId} to author's vector
follow / unfollow -> insert / erase in follower's hash set
         |
         ▼
getNewsFeed -> seed max-heap with newest tweet per visible author
         |
         ▼
pop newest -> emit tweetId -> push next-older from same author (repeat up to 10x)
```

- One heap slot per stream mirrors Merge K Sorted Lists, but traversed newest-to-oldest.
- Heap size stays bounded by `F + 1` regardless of total tweet volume.

* *Next Review Date:*
* *Key Takeaway:* When each source is already time-ordered, a max-heap over stream heads turns a full collect-and-sort into a lazy 10-step K-way Merge.
