# Classification Module

This module provides two main classifiers for text classification tasks: `YamlClassifier` and `RAGClassifier`. Both classifiers are designed to categorize user queries into predefined labels based on a set of examples and descriptions.

## YamlClassifier

The `YamlClassifier` is a simple yet powerful classifier that uses a YAML configuration file to define the classification task, labels, and examples.

### Features

- Load classification configuration from a YAML file
- Support for multiple labels with positive and negative examples
- Customizable number of examples to use for each label
- Synchronous and asynchronous prediction methods

## RAGClassifier

> Before running the `RAGClassifier` script, make sure to set an OPENAI_API_KEY variable in your shell.

The `RAGClassifier` is a classifier that uses a retreival model to classify user queries into predefined labels based on a set of examples and descriptions.

### Features

- Inherits all features from YamlClassifier
- Uses a vector database (ChromaDB) to store and retrieve similar examples
- Dynamically fetches similar examples for each query during classification
- Customizable number of similar examples to fetch (default: 2)
- Provides distance metrics for retrieved examples to aid in classification
- Supports fitting the classifier with examples and loading pre-fitted databases
