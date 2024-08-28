import json
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
import lancedb
from random import randint
from torch import nn
from collections import deque

db = lancedb.connect("../week1_bootstrap_evals/lancedb")
reviews_df = db.open_table("reviews").to_pandas()
reviews_df.id = reviews_df.id.astype(int)

with open("./ft_dataset.jsonl", "r") as f:
    finetune_data = [json.loads(line) for line in f]

neg_examples_per_q = 3

prepared_data = []
for item in finetune_data:
    query = item["question_with_context"]
    chunk_id = int(item["chunk_id"])

    # Fetch the relevant passage from the reviews table
    relevant_review = reviews_df.loc[chunk_id].review

    prepared_data.append(
        {
            "query": query,
            "positive_passage": relevant_review,
            "negative_passages": [],
        }
    )

    # Add negative examples
    for _ in range(neg_examples_per_q):
        random_chunk_id = randint(0, reviews_df.id.max())
        if random_chunk_id != chunk_id:
            random_passage = reviews_df.loc[random_chunk_id].review
            prepared_data[-1]["negative_passages"].append(random_passage)

# Prepare training examples
train_examples = []
for item in prepared_data:
    train_examples.append(
        InputExample(texts=[item["query"], item["positive_passage"]], label=1.0)
    )
    for negative_passage in item["negative_passages"]:
        train_examples.append(
            InputExample(texts=[item["query"], negative_passage], label=0.0)
        )

print(f"Created {len(train_examples)} training examples")

# Initialize the model
model = CrossEncoder("cross-encoder/stsb-distilroberta-base", num_labels=1)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Train the model
output_path = "./fine_tuned_reranker"
num_epochs = 1
print_loss_freq = 20


# Use custom loss function to help track loss while using SBERT
class CustomMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        # store losses to average when we print loss
        self.losses = deque(maxlen=print_loss_freq)
        self.batch_count = 0

    def forward(self, predictions, labels):
        loss = self.mse(predictions.view(-1), labels.view(-1))
        self.losses.append(loss.item())
        self.batch_count += 1

        if self.batch_count % print_loss_freq == 0:
            avg_loss = sum(self.losses) / len(self.losses)
            print(
                f"\nBatch {self.batch_count}, Average Loss (last {print_loss_freq} batches): {avg_loss:.4f}"
            )

        return loss


custom_loss = CustomMSELoss()

model.fit(
    train_dataloader=train_dataloader,
    epochs=num_epochs,
    warmup_steps=10,
    optimizer_params={"lr": 2e-5},
    show_progress_bar=True,
    loss_fct=custom_loss,
)

model.save(output_path)
print(f"Fine-tuning completed. Model saved to '{output_path}'.")
