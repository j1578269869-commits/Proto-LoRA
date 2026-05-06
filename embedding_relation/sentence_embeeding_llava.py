# import os
# import random
#
# import numpy as np
#
# from utils import weight_mean
#
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# from transformers import AutoTokenizer, AutoModel
# import json
# import torch
#
# def load_json_lines(filepath):
#     with open(filepath, 'r', encoding='utf-8') as file:
#         data = []                #data是一个列表，列表中的每个元素都是一个字典
#         print(filepath)
#         for line in file:
#             try:
#                 data.append(json.loads(line))
#             except json.JSONDecodeError as e:
#                 print(f"Error decoding JSON on line: {line}\nError: {e}")
#
#         return data
#
# def main():
#     filepath = 'data/my_refined_updated_relations.json'
#     data = load_json_lines(filepath)
#
#     # # 去掉 relation 字段
#     # for item in data:
#     #     if 'relation' in item:
#     #         del item['relation']
#     #
#     # print(data)
#     # 初始化模型和 tokenizer
#     model_name = '/home/NCUT/23/jz/AlignRE_LLaVA/vicuna-7b-v1.5'  # 使用正确的模型名称
#     token = 'os.getenv("HF_TOKEN")'
#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
#     model = AutoModel.from_pretrained(model_name, use_auth_token=token)
#
#
#     # 定义一个函数来生成 prompt 并获取句子表示
#     def get_sentence_embedding(item):
#         description_sentences_text = item['description']
#         label_name_sentences_text = item['label_name']
#
#         # aliases_sentences_text = ', '.join(item['aliases'])
#         aliases_sentences_text = item['aliases']
#         head_entity_type_sentences_text = ', '.join(item['h'])
#         tail_entity_type_sentences_text = ', '.join(item['t'])
#         aliases_sentences_text_0, aliases_sentences_text_1, aliases_sentences_text_2 = [], [], [] # 随机选取三个关系别名
#         random_id = random.sample(range(0, len(aliases_sentences_text)), 3)
#         aliases_sentences_text_0.append(aliases_sentences_text[random_id[0]])
#         aliases_sentences_text_1.append(aliases_sentences_text[random_id[1]])
#         aliases_sentences_text_2.append(aliases_sentences_text[random_id[2]])
#         aliases_sentences_text_0 = ', '.join(aliases_sentences_text_0)
#         aliases_sentences_text_1 = ', '.join(aliases_sentences_text_1)
#         aliases_sentences_text_2 = ', '.join(aliases_sentences_text_2)
#
#         description_sentences_text = tokenizer(description_sentences_text, return_tensors='pt')
#         label_name_sentences_text = tokenizer(label_name_sentences_text, return_tensors='pt')
#         aliases_sentences_text_0 = tokenizer(aliases_sentences_text_0, return_tensors='pt')
#         aliases_sentences_text_1 = tokenizer(aliases_sentences_text_1, return_tensors='pt')
#         aliases_sentences_text_2 = tokenizer(aliases_sentences_text_2, return_tensors='pt')
#
#         description_sentences_text = model(**description_sentences_text)
#         label_name_sentences_text = model(**label_name_sentences_text)
#         aliases_sentences_text_0 = model(**aliases_sentences_text_0)
#         aliases_sentences_text_1 = model(**aliases_sentences_text_1)
#         aliases_sentences_text_2 = model(**aliases_sentences_text_2)
#
#         description_sentences_text = description_sentences_text.last_hidden_state.mean(dim=1)
#         label_name_sentences_text = label_name_sentences_text.last_hidden_state.mean(dim=1)
#         aliases_sentences_text_0 = aliases_sentences_text_0.last_hidden_state.mean(dim=1)
#         aliases_sentences_text_1 = aliases_sentences_text_1.last_hidden_state.mean(dim=1)
#         aliases_sentences_text_2 = aliases_sentences_text_2.last_hidden_state.mean(dim=1)
#
#
#
#         return description_sentences_text,label_name_sentences_text,aliases_sentences_text_0,aliases_sentences_text_1,aliases_sentences_text_2
#
#
#     result = [get_sentence_embedding(item) for item in data]
#     description_sentence_embeddings, label_name_sentence_embeddings, aliases_name_sentence_embeddings_0, aliases_name_sentence_embeddings_1, aliases_name_sentence_embeddings_2 = zip(*result)
#
#     emb_dict = {}
#
#     # 获取每条数据的句子表示
#     for i, (emb_d, emb_l, emb_a0, emb_a1, emb_a2) in enumerate(
#             zip(description_sentence_embeddings, label_name_sentence_embeddings, aliases_name_sentence_embeddings_0,
#                 aliases_name_sentence_embeddings_1, aliases_name_sentence_embeddings_2)):
#         emb_dict[data[i]['relation']] = weight_mean([emb_d, emb_l, emb_a0, emb_a1, emb_a2]).tolist()
#
#
#     print("emb_dict:",type(emb_dict))
#     print("emb_dict的值的数据类型为",type(emb_dict.values()))
#
#     with open('similarity_embedding_idx.json', 'w', encoding='utf-8') as f:
#         json.dump(emb_dict, f)
#         print("saved successfully.")
#
#
#     emb_dict_list = list(emb_dict.values())
#     emb_dict = np.array(emb_dict_list)
#     emb_dict_list = torch.from_numpy(emb_dict).to(torch.float32)
#     print("emb_dict_list",type(emb_dict_list))
#     print("emb_dict_list",emb_dict_list.shape)
#     torch.save(emb_dict_list, 'embeddings_similarity.pt')
#     print("Embeddings saved successfully.")
#
#
#
#
#     # 保存 label_name 和对应的索引
#     label_name_to_idx = {item['label_name']: idx for idx, item in enumerate(data)}
#     # with open('label_name_to_embedding_idx.json', 'w', encoding='utf-8') as f:
#     #     json.dump(label_name_to_idx, f, ensure_ascii=False, indent=4)
#
#
#
#     print("label_name_to_idx",label_name_to_idx)        #label_name_to_idx {'place of birth': 0, 'parent': 1, 'couple': 2, 'nationality': 3, 'member of': 14, 'alumi': 5, 'locate at': 7, 'religion': 8, 'awarded': 9, 'neighbor': 10, 'held on': 11, 'subsidiary': 12, 'part of': 13, 'place of residence': 15, 'present in': 16, 'charges': 17, 'contain': 18, 'peer': 19, 'alternate names': 20, 'race': 21, 'siblings': 22, 'none': 23}
#     print("label_name_to_idx",type(label_name_to_idx))   #label_name_to_idx <class 'dict'>
#     # 如果需要保存 embeddings，可以使用 torch.save
#     # torch.save(embeddings, 'data/embeddings222.pt')
#     # print("Embeddings saved successfully.")
# def weight_mean(sentence_vectors):   #用于计算输入向量列表的加权平均值
#     def cosine_similarity_(v1, v2):
#         v1 = v1.detach().numpy().flatten()
#         v2 = v2.detach().numpy().flatten()
#         dot_product = np.dot(v1, v2)
#         norm_v1 = np.linalg.norm(v1)
#         norm_v2 = np.linalg.norm(v2)
#         similarity = dot_product / (norm_v1 * norm_v2)
#         return similarity
#
#     num_sentences = len(sentence_vectors)
#
#     similarity_matrix = np.zeros((num_sentences, num_sentences))
#     for i in range(num_sentences):
#         for j in range(num_sentences):
#             similarity_matrix[i, j] = cosine_similarity_(sentence_vectors[i], sentence_vectors[j])
#
#     sentence_weights = np.mean(similarity_matrix, axis=1)
#
#     sentence_vectors = [vec.detach().numpy() for vec in sentence_vectors]  #新加 ，将tensor转换为numpy数组
#     merged_vector = np.average(sentence_vectors, axis=0, weights=sentence_weights)
#
#     return merged_vector
#
#
# if __name__ == "__main__":
#     main()

# import os
# import random
# import json
# import torch
# import numpy as np
# from transformers import AutoTokenizer, AutoModel
#
# def load_json_lines(filepath):
#     """Load JSON lines from a file."""
#     with open(filepath, 'r', encoding='utf-8') as file:
#         data = []
#         for line in file:
#             try:
#                 data.append(json.loads(line))
#             except json.JSONDecodeError as e:
#                 print(f"Error decoding JSON on line: {line}\nError: {e}")
#         return data
#
# def get_random_aliases(aliases, num=3):
#     """Randomly select `num` aliases from the list of aliases."""
#     if len(aliases) < num:
#         raise ValueError("Not enough aliases to sample.")
#     return random.sample(aliases, num)
#
# def tokenize_and_encode(texts, tokenizer):
#     """Tokenize and encode texts using the provided tokenizer."""
#     return [tokenizer(text, return_tensors='pt') for text in texts]
#
# def get_sentence_embedding(item, tokenizer, model):
#     """Generate sentence embeddings for various parts of an item."""
#     description = item['description']
#     label_name = item['label_name']
#     aliases = get_random_aliases(item['aliases'])
#     head_entities = ', '.join(item['h'])
#     tail_entities = ', '.join(item['t'])
#
#     texts = [description, label_name] + aliases
#     encoded_texts = tokenize_and_encode(texts, tokenizer)
#     embeddings = [model(**text).last_hidden_state.mean(dim=1) for text in encoded_texts]
#
#     return tuple(embeddings)
#
# def weight_mean(sentence_vectors):
#     """Compute the weighted mean of sentence vectors based on cosine similarity."""
#     def cosine_similarity(v1, v2):
#         v1 = v1.detach().numpy().flatten()
#         v2 = v2.detach().numpy().flatten()
#         dot_product = np.dot(v1, v2)
#         norm_v1 = np.linalg.norm(v1)
#         norm_v2 = np.linalg.norm(v2)
#         return dot_product / (norm_v1 * norm_v2)
#
#     num_sentences = len(sentence_vectors)
#     similarity_matrix = np.array([[cosine_similarity(v1, v2) for v2 in sentence_vectors] for v1 in sentence_vectors])
#     sentence_weights = np.mean(similarity_matrix, axis=1)
#     sentence_vectors_np = [vec.detach().numpy() for vec in sentence_vectors]
#     merged_vector = np.average(sentence_vectors_np, axis=0, weights=sentence_weights)
#
#     return merged_vector
#
# def main():
#     filepath = 'data/my_refined_updated_relations.json'
#     data = load_json_lines(filepath)
#
#     model_name = '/home/NCUT/23/jz/AlignRE_LLaVA/vicuna-7b-v1.5'
#     token = 'os.getenv("HF_TOKEN")'
#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
#     model = AutoModel.from_pretrained(model_name, use_auth_token=token)
#
#     embeddings = [get_sentence_embedding(item, tokenizer, model) for item in data]
#     emb_dict = {
#         item['relation']: weight_mean(embedding).tolist()
#         for item, embedding in zip(data, embeddings)
#     }
#
#     with open('similarity_embedding_idx.json', 'w', encoding='utf-8') as f:
#         json.dump(emb_dict, f)
#         print("Embeddings saved successfully.")
#
#     emb_tensor = torch.tensor(list(emb_dict.values()), dtype=torch.float32)
#     torch.save(emb_tensor, 'embeddings_similarity20250511.pt')
#     print("PyTorch tensor saved successfully.")
#     # 在main()函数的最后添加：
#
#     print("emb_dict的值的数据类型为", type(emb_dict.values()))
#     print(f"emb_dict.keys(): {len(emb_dict.keys())}")
#     print("emb_dict.values():", len(emb_dict.values()))
#
#     label_name_to_idx = {item['label_name']: idx for idx, item in enumerate(data)}
#     print("Label name to index mapping:", label_name_to_idx)
#
# if __name__ == "__main__":
#     main()


import os
import random
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

def load_json_lines(filepath):
    """Load JSON lines from a file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        data = []
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line}\nError: {e}")
        return data

def get_random_aliases(aliases, num=3):
    """Randomly select `num` aliases from the list of aliases."""
    if len(aliases) < num:
        raise ValueError("Not enough aliases to sample.")
    return random.sample(aliases, num)

def tokenize_and_encode(texts, tokenizer):
    """Tokenize and encode texts using the provided tokenizer."""
    return [tokenizer(text, return_tensors='pt') for text in texts]

def get_sentence_embedding(item, tokenizer, model):
    """Generate sentence embeddings for label_name of an item."""
    label_name = item['label_name']
    texts = [label_name]
    encoded_texts = tokenize_and_encode(texts, tokenizer)
    embeddings = [model(**text).last_hidden_state.mean(dim=1) for text in encoded_texts]

    return tuple(embeddings)

def weight_mean(sentence_vectors):
    """Compute the weighted mean of sentence vectors based on cosine similarity."""
    def cosine_similarity(v1, v2):
        v1 = v1.detach().numpy().flatten()
        v2 = v2.detach().numpy().flatten()
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)

    num_sentences = len(sentence_vectors)
    similarity_matrix = np.array([[cosine_similarity(v1, v2) for v2 in sentence_vectors] for v1 in sentence_vectors])
    sentence_weights = np.mean(similarity_matrix, axis=1)
    sentence_vectors_np = [vec.detach().numpy() for vec in sentence_vectors]
    merged_vector = np.average(sentence_vectors_np, axis=0, weights=sentence_weights)

    return merged_vector

def main():
    filepath = 'my_refined_updated_relations.json'
    data = load_json_lines(filepath)

    model_name = '/home/NCUT/23/jz/AlignRE_LLaVA/vicuna-7b-v1.5'
    token = 'os.getenv("HF_TOKEN")'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = AutoModel.from_pretrained(model_name, use_auth_token=token)

    embeddings = [get_sentence_embedding(item, tokenizer, model) for item in data]
    emb_dict = {
        item['relation']: weight_mean(embedding).tolist()
        for item, embedding in zip(data, embeddings)
    }

    with open('similarity_embedding_idx_llava.json', 'w', encoding='utf-8') as f:
        json.dump(emb_dict, f)
        print("Embeddings saved successfully.")

    emb_tensor = torch.tensor(list(emb_dict.values()), dtype=torch.float32)
    torch.save(emb_tensor, 'embeddings_similarity_label_llava.pt')
    print("PyTorch tensor saved successfully.")


    label_name_to_idx = {item['label_name']: idx for idx, item in enumerate(data)}
    print("Label name to index mapping:", label_name_to_idx)

if __name__ == "__main__":
    main()
