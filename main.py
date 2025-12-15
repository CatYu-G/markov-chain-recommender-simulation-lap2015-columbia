"""
Simulating Personalized Content Recommendations with Markov Chains

This script simulates a personalized recommender system, models the sequence
of recommended categories as a Markov chain, and analyzes long-run behavior
using stationary distributions and entropy.

Author: Yu Gu et al.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
#  SECTION 1 — GLOBAL SETUP
# ============================================================

# Content categories
CATEGORIES = ["Sports", "Fashion", "Memes", "Politics", "Music"]
K = len(CATEGORIES)

# True (latent) user preferences (fixed, hidden from the algorithm)
TRUE_PREF = np.array([0.1, 0.3, 0.4, 0.05, 0.15], dtype=float)
TRUE_PREF = TRUE_PREF / TRUE_PREF.sum()  # normalize

# Random seed for reproducibility
np.random.seed(42)


# ============================================================
#  SECTION 2 — CORE RECOMMENDER MECHANICS
# ============================================================

def choose_category(belief: np.ndarray) -> int:
    """
    Sample a category according to the current belief distribution.
    """
    return np.random.choice(len(belief), p=belief)


def generate_rating(category_idx: int, true_pref: np.ndarray) -> int:
    """
    Generate a user rating (1–5) based on true preferences with noise.
    """
    expected = 1 + 4 * true_pref[category_idx]
    noisy = expected + np.random.normal(0, 0.3)
    return int(np.clip(round(noisy), 1, 5))


def update_belief(
    belief: np.ndarray,
    category_idx: int,
    rating: int,
    alpha: float
) -> np.ndarray:
    """
    Update the belief vector after observing a rating.
    """
    new_belief = belief.copy()

    # Shift belief based on rating relative to neutral value (3)
    delta = (rating - 3) * alpha
    new_belief[category_idx] += delta

    # Prevent invalid probabilities
    new_belief = np.clip(new_belief, 0.001, None)
    new_belief /= new_belief.sum()

    return new_belief


# ============================================================
#  SECTION 3 — FULL SIMULATION
# ============================================================

def run_simulation(
    alpha: float,
    T: int,
    true_pref: np.ndarray
):
    """
    Run the recommendation process for T steps.

    Returns:
        categories_seen: list[int]
        ratings: list[int]
        final_belief: np.ndarray
    """
    belief = np.ones(K) / K
    categories_seen = []
    ratings = []

    for _ in range(T):
        cat = choose_category(belief)
        rating = generate_rating(cat, true_pref)

        categories_seen.append(cat)
        ratings.append(rating)

        belief = update_belief(belief, cat, rating, alpha)

    return categories_seen, ratings, belief


# ============================================================
#  SECTION 4 — MARKOV CHAIN ANALYSIS
# ============================================================

def build_transition_matrix(categories_seen: list[int]) -> np.ndarray:
    """
    Construct the empirical Markov transition matrix.
    """
    counts = np.zeros((K, K))

    for t in range(len(categories_seen) - 1):
        i = categories_seen[t]
        j = categories_seen[t + 1]
        counts[i, j] += 1

    P = np.zeros_like(counts)
    for i in range(K):
        if counts[i].sum() > 0:
            P[i] = counts[i] / counts[i].sum()
        else:
            P[i] = np.ones(K) / K

    return P


def stationary_distribution(P: np.ndarray, iterations: int = 5000) -> np.ndarray:
    """
    Compute the stationary distribution using power iteration.
    """
    v = np.ones(K) / K
    for _ in range(iterations):
        v = v @ P
    return v


# ============================================================
#  SECTION 5 — METRICS
# ============================================================

def entropy(dist: np.ndarray) -> float:
    """
    Shannon entropy of a probability distribution.
    """
    dist = dist[dist > 0]
    return -np.sum(dist * np.log(dist))


# ============================================================
#  SECTION 6 — EXPERIMENT: ALPHA SWEEP
# ============================================================

def alpha_sweep(
    alphas: list[float],
    T: int
):
    """
    Run the full experiment across multiple personalization strengths.
    """
    results = []

    for alpha in alphas:
        cats, ratings, _ = run_simulation(alpha, T, TRUE_PREF)
        P = build_transition_matrix(cats)
        pi = stationary_distribution(P)

        H = entropy(pi)
        avg_rating = np.mean(ratings)

        results.append({
            "alpha": alpha,
            "entropy": H,
            "avg_rating": avg_rating,
            "stationary": pi
        })

    return results


# ============================================================
#  SECTION 7 — PLOTTING
# ============================================================

def plot_results(results):
    alphas = [r["alpha"] for r in results]
    entropies = [r["entropy"] for r in results]
    ratings = [r["avg_rating"] for r in results]

    # Entropy plot
    plt.figure()
    plt.plot(alphas, entropies, marker="o")
    plt.xlabel("Personalization strength (alpha)")
    plt.ylabel("Entropy (diversity)")
    plt.title("Diversity vs Personalization")
    plt.grid(True)
    plt.savefig("entropy_vs_alpha.png")

    # Rating plot
    plt.figure()
    plt.plot(alphas, ratings, marker="o")
    plt.xlabel("Personalization strength (alpha)")
    plt.ylabel("Average rating")
    plt.title("User Satisfaction vs Personalization")
    plt.grid(True)
    plt.savefig("rating_vs_alpha.png")


# ============================================================
#  SECTION 8 — MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    ALPHAS = [0.0, 0.1, 0.3, 0.7, 1.0]
    T = 5000

    results = alpha_sweep(ALPHAS, T)

    print("Alpha | Entropy | Avg Rating | Stationary Distribution")
    for r in results:
        print(
            f"{r['alpha']:>4} | "
            f"{r['entropy']:.4f} | "
            f"{r['avg_rating']:.4f} | "
            f"{np.round(r['stationary'], 3)}"
        )

    plot_results(results)
