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
The generator GG aims to produce challenging queries that maximize the retriever's loss, while being relevant. The objective function for the generator is:

$$\max_G \mathbb{E}_{q \sim G}[\mathcal{L}_R(q, R(q)) + S(q)]$$
where S(q) is a relevance score that evaluates the relevance of the generated queries.

## Combined Framework
The interaction between the generator and the retriever ensures that the retriever is constantly challenged and improved, while the generator is guided to produce high-quality challenging queries.

In this experiment, Retriever is trained as a binary classifier to classify a given pair of (query, document) as positive vs negative. The model is initialized with google's BERT large and trained to minimize the retriever's objective mentioned above.
The positive class score can be used for search ranking or retrieval.

Generator is trained through RL using policy gradient algorithm. The Generator's objective mentioned above is treated as the reward. The model is initialized with google's T5 model (text-2-text) and tuned to maximize the reward.

## Dataset and Trained models
The models are trained end-to-end using [Wayfair annotation dataset](https://github.com/wayfair/WANDS).

Trained models are available on huggingface:
[Generator](https://huggingface.co/prhegde/search-query-generator-ecommerce) and [Retriever](https://huggingface.co/prhegde/query-product-relevance-model-ecommerce)


