# Concepts

## Human-Relevance Question

A checklist question identified as relevant to a sample by human-relevance
annotation or by a model-generated human-relevance extractor intended to mimic
human reasoning.

## Effective Accuracy

The fraction of all examples answered correctly when ties are treated as
abstentions:

`effective_accuracy = accuracy_on_non_ties * n_valid / n_total`.

## Human-Relevance Oracle

An oracle selector that ranks or selects questions from human-relevance
annotations rather than from the learned selector scores. It is used as an upper
bound for the selector retrieval problem, not as a deployable model.
