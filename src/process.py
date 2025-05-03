from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    return model.encode(text, convert_to_tensor=True)

def get_similarity(tensor1, tensor2):
    return util.cos_sim(tensor1, tensor2).item()