import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ISLP import load_data

# -----------------------------
# 1. Load dataset (NO INTERNET)
# -----------------------------
USArrests = load_data("USArrests")

# -----------------------------
# 2. Standardize data
# -----------------------------
scaler = StandardScaler(with_std=True, with_mean=True)
USArrests_scaled = scaler.fit_transform(USArrests)

# -----------------------------
# 3. Apply PCA
# -----------------------------
pcaUS = PCA()
pcaUS.fit(USArrests_scaled)

scores = pcaUS.transform(USArrests_scaled)

# -----------------------------
# 4. Biplot (PC1 vs PC2)
# -----------------------------
i, j = 0, 1

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(scores[:, i], scores[:, j])

ax.set_xlabel(f'PC{i+1}')
ax.set_ylabel(f'PC{j+1}')

# arrows (loadings)
for k in range(pcaUS.components_.shape[1]):
    ax.arrow(0, 0,
             pcaUS.components_[i, k],
             pcaUS.components_[j, k],
             color='r')
    ax.text(pcaUS.components_[i, k],
            pcaUS.components_[j, k],
            USArrests.columns[k])

ax.set_title("PCA Biplot (Before Flip)")
plt.grid()

# -----------------------------
# 5. Flip PC2 (for better orientation)
# -----------------------------
scores[:, 1] *= -1
pcaUS.components_[1] *= -1

# scaled arrows
s_ = 2

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(scores[:, i], scores[:, j])

ax.set_xlabel(f'PC{i+1}')
ax.set_ylabel(f'PC{j+1}')

for k in range(pcaUS.components_.shape[1]):
    ax.arrow(0, 0,
             s_ * pcaUS.components_[i, k],
             s_ * pcaUS.components_[j, k],
             color='r')
    ax.text(s_ * pcaUS.components_[i, k],
            s_ * pcaUS.components_[j, k],
            USArrests.columns[k])

ax.set_title("PCA Biplot (After Flip)")
plt.grid()

# -----------------------------
# 6. Variance explained
# -----------------------------
print("\nExplained Variance:")
print(pcaUS.explained_variance_)

print("\nExplained Variance Ratio:")
print(pcaUS.explained_variance_ratio_)

print("\nCumulative Variance:")
print(np.cumsum(pcaUS.explained_variance_ratio_))

# -----------------------------
# 7. Scree plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ticks = np.arange(1, pcaUS.n_components_ + 1)

# individual variance
axes[0].plot(ticks, pcaUS.explained_variance_ratio_, marker='o')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Proportion of Variance Explained')
axes[0].set_xticks(ticks)
axes[0].set_ylim([0, 1])
axes[0].set_title("Scree Plot")

# cumulative variance
axes[1].plot(ticks, np.cumsum(pcaUS.explained_variance_ratio_), marker='o')
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Cumulative Variance Explained')
axes[1].set_xticks(ticks)
axes[1].set_ylim([0, 1])
axes[1].set_title("Cumulative Variance")

plt.tight_layout()
plt.show()

# -----------------------------
# 8. Check relation
# -----------------------------
print("\nStd dev of scores:")
print(scores.std(axis=0, ddof=1))

print("\nSqrt of eigenvalues:")
print(np.sqrt(pcaUS.explained_variance_))

# -----------------------------
# 9. Extra (cumsum example)
# -----------------------------
a = np.array([1, 2, 8, -3])
print("\nCumulative sum example:", np.cumsum(a))