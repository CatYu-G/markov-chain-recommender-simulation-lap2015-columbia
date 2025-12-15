# Simulating Personalized Content Recommendations with Markov Chains

This project simulates a personalized content recommendation system and studies
how personalization strength affects long-run content diversity and user satisfaction.

We model the sequence of recommended content categories as a **Markov chain**,
estimate its transition matrix from simulation data, and analyze the **stationary distribution**
and **entropy** to quantify filter-bubble effects.

---

## Core Idea

- A recommender maintains a belief distribution over content categories.
- At each step, it recommends a category, receives a user rating (1–5),
  and updates its belief based on a personalization parameter α.
- The resulting sequence of categories is treated as a Markov chain.

We study:
- **Stationary distribution** → long-run exposure
- **Entropy** → content diversity
- **Average rating** → user satisfaction

---

## Key Results

- No personalization (α = 0): high diversity, low satisfaction
- Moderate personalization (α ≈ 0.1): **strongest filter bubble**, highest satisfaction
- High personalization (α → 1): unstable behavior, diversity increases again

This reveals a **non-monotonic trade-off** between diversity and engagement.

---

## Output

Alpha | Entropy | Avg Rating | Stationary Distribution
 0.0 | 1.6086 | 1.7954 | [0.209 0.202 0.207 0.188 0.194]
 0.1 | 0.2969 | 2.5808 | [0.012 0.018 0.943 0.013 0.014]
 0.3 | 0.6424 | 2.4872 | [0.033 0.056 0.845 0.033 0.034]
 0.7 | 1.3475 | 2.1544 | [0.106 0.158 0.522 0.101 0.114]
 1.0 | 1.5217 | 1.9936 | [0.15  0.174 0.38  0.148 0.147]

 

