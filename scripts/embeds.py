from openai import OpenAI
from datasets import load_dataset, Dataset,load_from_disk
import numpy as np
import matplotlib.pyplot as plt


client = OpenAI(api_key)


personaMap = {}
persona_num = 6

for i in range(persona_num):
    personaMap[i] = load_from_disk(f"./data/personas/{i}")["persona"][0]

print(personaMap)
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=list(personaMap.values())
)

embeddings = [np.array(item.embedding) for item in response.data]

# Normalize each vector to unit length
X_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Compute cosine similarity matrix
cosine_sim = X_norm @ X_norm.T

M = cosine_sim

print(np.sum((M - np.eye(M.shape[0])), axis=1))

plt.imshow(cosine_sim, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.savefig("embed.png") 