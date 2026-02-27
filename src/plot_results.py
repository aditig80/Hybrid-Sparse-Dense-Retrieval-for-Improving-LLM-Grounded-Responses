import matplotlib.pyplot as plt

# Alpha values and corresponding Recall@5
alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
recall_scores = [0.394, 0.466, 0.554, 0.674, 0.789, 0.779]

plt.figure()
plt.plot(alphas, recall_scores, marker='o')

plt.xlabel("Alpha (Dense Weight)")
plt.ylabel("Recall@5")
plt.title("Hybrid Retrieval: Alpha vs Recall@5")

plt.xticks(alphas)
plt.ylim(0, 1)

plt.grid(True)

plt.savefig("results/alpha_vs_recall.png")
plt.show()