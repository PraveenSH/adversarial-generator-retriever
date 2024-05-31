# adversarial-generator-retriever

## Introduction
Effective document retrieval is a critical task in various applications, including search engines, recommendation systems, and information retrieval. However, one of the main challenges in building robust retrieval systems is the quality and diversity of training data. Traditional approaches often struggle with limited and biased data, leading to suboptimal performance.

To address these challenges, we propose a novel generator-retriever framework where a generator creates challenging queries to improve the retriever's performance. This approach ensures that the retriever is exposed to a wide range of scenarios, enabling it to learn more effectively and generalize better to unseen data.

## Problem Formulation
### Retriever Objective
The goal of the retriever **R** is to correctly classify query-document pairs as positive (relevant) or negative (irrelevant). The objective function for the retriever is to minimize the expected loss over the generated queries and documents:
$$\min_R\mathbb{E}_ {\mathcal{(q,d) \in D}}[\mathcal{L}_R(q,d)]$$
where $\mathcal{L}_R$ is the binary cross-entropy loss.

## Generator Objective
The generator GG aims to produce challenging queries that maximize the retriever's loss, weighted by their relevance. The objective function for the generator is:

$$\max_G \mathbb{E}_{q \sim G}[\mathcal{L}_R(q, R(q)) + S(q)]$$
where S(q) is a relevance score that evaluates the relevance of the generated queries.

## Combined Framework
The interaction between the generator and the retriever ensures that the retriever is constantly challenged and improved, while the generator is guided to produce high-quality challenging queries.

The models are trained end-to-end using [Wayfair annotation dataset](https://github.com/wayfair/WANDS)
