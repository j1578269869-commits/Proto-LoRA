import os
import random

import numpy as np

from utils import weight_mean

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
import json
import torch



def weight_mean(sentence_vectors):
    """
    Compute the weighted mean of sentence vectors based on cosine similarity.
    Optimized for cases where there is only one vector in sentence_vectors.

    Args:
        sentence_vectors (list of torch.Tensor): List of sentence vectors.

    Returns:
        list: Weighted mean vector as a list.
    """
    # If there's only one vector, directly return it as the weighted mean
    if len(sentence_vectors) == 1:
        return sentence_vectors[0].detach().cpu().numpy().flatten().tolist()

    # Convert to numpy array for efficient computation
    sentence_vectors_np = torch.stack(sentence_vectors).cpu().detach().numpy()

    # Compute cosine similarity matrix
    def cosine_similarity_matrix(vectors):
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / norm
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        return similarity_matrix

    similarity_matrix = cosine_similarity_matrix(sentence_vectors_np)

    # Compute sentence weights as the mean similarity of each sentence
    sentence_weights = np.mean(similarity_matrix, axis=1)

    # Compute the weighted mean of sentence vectors
    merged_vector = np.average(sentence_vectors_np, axis=0, weights=sentence_weights)

    return merged_vector
def load_json_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = []                #data是一个列表，列表中的每个元素都是一个字典
        print(filepath)
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line}\nError: {e}")

        return data

def main():
    filepath = 'data/my_refined_updated_entities.json'
    data = load_json_lines(filepath)

    # # 去掉 relation 字段
    # for item in data:
    #     if 'relation' in item:
    #         del item['relation']
    #
    # print(data)
    # 初始化模型和 tokenizer
    model_name = '/home/NCUT/23/jz/Qwen2.5-7B-Instruct'   # 使用正确的模型名称
    token = 'os.getenv("HF_TOKEN")'
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    # model = AutoModel.from_pretrained(model_name, use_auth_use_auth_token=token,torch_dtype=torch.float16,device_map="auto")
    # model = AutoModel.from_pretrained(model_name, use_auth_token=token,torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = AutoModel.from_pretrained(model_name, use_auth_token=token,torch_dtype=torch.float16)


    # 定义一个函数来生成 prompt 并获取句子表示
    # def get_sentence_embedding(item):
    #     description_sentences_text = item['description']
    #     label_name_sentences_text = item['label_name']
    #
    #     # aliases_sentences_text = ', '.join(item['aliases'])
    #     aliases_sentences_text = item['aliases']
    #     head_entity_type_sentences_text = ', '.join(item['h'])
    #     tail_entity_type_sentences_text = ', '.join(item['t'])
    #     aliases_sentences_text_0, aliases_sentences_text_1, aliases_sentences_text_2 = [], [], [] # 随机选取三个关系别名
    #     random_id = random.sample(range(0, len(aliases_sentences_text)), 3)
    #     aliases_sentences_text_0.append(aliases_sentences_text[random_id[0]])
    #     aliases_sentences_text_1.append(aliases_sentences_text[random_id[1]])
    #     aliases_sentences_text_2.append(aliases_sentences_text[random_id[2]])
    #     aliases_sentences_text_0 = ', '.join(aliases_sentences_text_0)
    #     aliases_sentences_text_1 = ', '.join(aliases_sentences_text_1)
    #     aliases_sentences_text_2 = ', '.join(aliases_sentences_text_2)
    #
    #     description_sentences_text = tokenizer(description_sentences_text, return_tensors='pt')
    #     label_name_sentences_text = tokenizer(label_name_sentences_text, return_tensors='pt')
    #     aliases_sentences_text_0 = tokenizer(aliases_sentences_text_0, return_tensors='pt')
    #     aliases_sentences_text_1 = tokenizer(aliases_sentences_text_1, return_tensors='pt')
    #     aliases_sentences_text_2 = tokenizer(aliases_sentences_text_2, return_tensors='pt')
    #
    #     description_sentences_text = model.encoder(**description_sentences_text)
    #     label_name_sentences_text = model.encoder(**label_name_sentences_text)
    #     aliases_sentences_text_0 = model.encoder(**aliases_sentences_text_0)
    #     aliases_sentences_text_1 = model.encoder(**aliases_sentences_text_1)
    #     aliases_sentences_text_2 = model.encoder(**aliases_sentences_text_2)
    #
    #     # description_sentences_text =  model.decoder(**description_sentences_text)
    #     # label_name_sentences_text =  model.decoder(**label_name_sentences_text)
    #     # aliases_sentences_text_0 =  model.decoder(**aliases_sentences_text_0)
    #     # aliases_sentences_text_1 =  model.decoder(**aliases_sentences_text_1)
    #     # aliases_sentences_text_2 =  model.decoder(**aliases_sentences_text_2)
    #     #现在是编码器的last_hidden_state
    #     description_sentences_text = description_sentences_text.last_hidden_state.mean(dim=1)
    #     label_name_sentences_text = label_name_sentences_text.last_hidden_state.mean(dim=1)
    #     aliases_sentences_text_0 = aliases_sentences_text_0.last_hidden_state.mean(dim=1)
    #     aliases_sentences_text_1 = aliases_sentences_text_1.last_hidden_state.mean(dim=1)
    #     aliases_sentences_text_2 = aliases_sentences_text_2.last_hidden_state.mean(dim=1)
    #
    #     return description_sentences_text,label_name_sentences_text,aliases_sentences_text_0,aliases_sentences_text_1,aliases_sentences_text_2
    #
    # result = [get_sentence_embedding(item) for item in data]
    # description_sentence_embeddings, label_name_sentence_embeddings, aliases_name_sentence_embeddings_0, aliases_name_sentence_embeddings_1, aliases_name_sentence_embeddings_2 = zip(*result)
    #
    # emb_dict = {}
    #
    # # 获取每条数据的句子表示
    # for i, (emb_d, emb_l, emb_a0, emb_a1, emb_a2) in enumerate(
    #         zip(description_sentence_embeddings, label_name_sentence_embeddings, aliases_name_sentence_embeddings_0,
    #             aliases_name_sentence_embeddings_1, aliases_name_sentence_embeddings_2)):
    #     emb_dict[data[i]['relation']] = weight_mean([emb_d, emb_l, emb_a0, emb_a1, emb_a2]).tolist()
    def get_sentence_embedding(item):
        # 提取 label_name 并将其转换为句子嵌入
        print(item['label_name'])
        label_name_sentences_text = item['label_name']

        # 使用 tokenizer 将文本转换为模型输入格式
        label_name_sentences_text = tokenizer(label_name_sentences_text, return_tensors='pt')

        # 使用模型的 encoder 获取嵌入表示
        label_name_sentences_text = model(**label_name_sentences_text)

        # 对 last_hidden_state 取平均值作为句子嵌入
        label_name_sentences_text = label_name_sentences_text.last_hidden_state.mean(dim=1)

        return label_name_sentences_text

    # 遍历数据并获取每个 item 的 label_name 句子嵌入
    result = [get_sentence_embedding(item) for item in data]

    # 将结果解包为 label_name_sentence_embeddings
    label_name_sentence_embeddings = [emb for emb in result]

    # 构建嵌入字典
    emb_dict = {
        data[i]['label_name']: weight_mean([emb])
        for i, emb in enumerate(label_name_sentence_embeddings)
    }

    print("emb_dict:",type(emb_dict))
    print("emb_dict的值的数据类型为",type(emb_dict.values()))

    with open('entity_embedding_idx_Qwen2.json', 'w', encoding='utf-8') as f:
        json.dump(emb_dict, f)
        print("saved successfully.")


    emb_dict_list = list(emb_dict.values())
    emb_dict = np.array(emb_dict_list)
    emb_dict_list = torch.from_numpy(emb_dict).to(torch.float32)
    print("emb_dict_list",type(emb_dict_list))
    print("emb_dict_list",emb_dict_list.shape)
    torch.save(emb_dict_list, 'data/entity_embedding_idx_Qwen2.pt')
    print("Embeddings saved successfully.")




    # 保存 label_name 和对应的索引
    label_name_to_idx = {item['label_name']: idx for idx, item in enumerate(data)}
    # with open('label_name_to_embedding_idx.json', 'w', encoding='utf-8') as f:
    #     json.dump(label_name_to_idx, f, ensure_ascii=False, indent=4)



    print("label_name_to_idx",label_name_to_idx)
    print("label_name_to_idx",type(label_name_to_idx))   #label_name_to_idx <class 'dict'>
    # 如果需要保存 embeddings，可以使用 torch.save
    # torch.save(embeddings, 'data/embeddings222.pt')
    # print("Embeddings saved successfully.")


if __name__ == "__main__":
    main()

