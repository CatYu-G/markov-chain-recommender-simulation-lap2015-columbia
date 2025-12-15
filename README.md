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

## Run the code and check it out by yourself

 

