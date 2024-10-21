import chromadb
from jinja2 import Template
from yaml_classifier import YamlClassifier
from textwrap import dedent
from pydantic import Field
import chromadb.utils.embedding_functions as embedding_functions
import os
from pydantic import Field
import instructor
import openai


class RAGClassifier(YamlClassifier):
    fetch_n_examples: int | None = Field(default=2)
    __db = None

    def load_db(self, collection_name: str):
        if collection_name:
            chroma_client = chromadb.Client()
            self.__db = chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.get_embedding_function(),
            )
        return self

    def get_embedding_function(self):
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )

    def fit(self, collection_name: str):
        chroma_client = chromadb.Client()

        self.__db = chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=self.get_embedding_function()
        )

        all_examples = []
        for label in self.labels:
            for example in label.examples.positive:
                all_examples.append({"text": example, "label": label.name})
            for example in label.examples.negative:
                all_examples.append({"text": example, "label": label.name})

        # Upsert all examples into the database
        self.__db.upsert(
            documents=[example["text"] for example in all_examples],
            ids=[str(i) for i in range(len(all_examples))],
            metadatas=[{"label": example["label"]} for example in all_examples],
        )

    def get_user_query(self, query: str) -> str:
        if self.__db is None:
            raise ValueError("Database not initialized, call cls.fit() first")
        results = self.__db.query(
            query_texts=[query],
            n_results=self.fetch_n_examples,
        )
        formatted_results = [
            (doc, metadata["label"], distance)
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

        template = Template(
            dedent(
                """
        Classify the following document:

        <doc>
        {{ query }}
        </doc>

        Similar examples:
        <examples>
        {% for doc, label, distance in formatted_results %}
        <example>
            <distance> {{ "%.2f"|format(distance) }} </distance>
            <label> {{ label }} </label>
            <similar_document> {{ doc }} </similar_document>
        </example>
        {% endfor %}
        </examples>

        Provide your classification based on the above information.
        """
            )
        )

        return template.render(query=query, formatted_results=formatted_results)


if __name__ == "__main__":
    client = instructor.from_openai(openai.OpenAI())

    classifier = RAGClassifier.load("example.yaml")
    classifier.fit("example")
    # classifier.load_db("example")

    print("# Example of User Query")
    print(
        classifier.get_user_query(
            "When can i expect to see the next episode of the show?"
        )
    )
