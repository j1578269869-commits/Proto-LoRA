import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, get_linear_schedule_with_warmup, \
    Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import json
import os
from argparse import ArgumentParser

# 设置设备
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=25)
args = parser.parse_args()

model_name = "/home/NCUT/23/jz/Qwen2-VL-7B"

# 加载 tokenizer 和 processor（需 trust_remote_code）
# tokenizer = Qwen2VLProcessor.from_pretrained(model_name, trust_remote_code=True)
# processor = Qwen2VLProcessor.from_pretrained(model_name)
processor = Qwen2VLProcessor.from_pretrained(model_name)
def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# ======================
# 关系映射表（同原代码）
# ======================
relation_map = {
    "P19": "place_of_birth",
    "P22": "parent",
    "P26": "couple",
    "P27": "nationality",
    "P31": "alternate_names_org",
    "P69": "alumi",
    "P102": "member_of",
    "P131": "locate_at",
    "P140": "religion",
    "P166": "awarded",
    "P47": "neighbor",
    "P276": "held_on",
    "P355": "subsidiary",
    "P361": "part_of",
    "P463": "member_of",
    "P551": "place_of_residence",
    "P1344": "present_in",
    "P2389": "charges",
    "P527": "contain",
    "P188": "peer",
    "P259": "alternate_names_per",
    "P99": "race",
    "P685": "siblings",
    "P0": "none"
}

# 候选关系列表（含标准化小写）
candidate_relations = [
    "place_of_birth",
    "parent",
    "couple",
    "nationality",
    "alternate_names_org",
    "alumi",
    "member_of",
    "locate_at",
    "religion",
    "awarded",
    "neighbor",
    "held_on",
    "subsidiary",
    "part_of",
    "member_of",
    "place_of_residence",
    "present_in",
    "charges",
    "contain",
    "peer",
    "alternate_names_per",
    "race",
    "siblings",
    "none"
]

rel_lower = {rel.lower(): rel for rel in candidate_relations}  # 标准化映射

options_str = "Options:\n" + "\n".join([f"{i + 1}. {rel}" for i, rel in enumerate(candidate_relations)])
def split_types(types):
    # 过滤掉不需要的类型。
    filtered_types = ["Other", "None", "none", "NA","P0"]
    types = [ele for ele in types if ele not in filtered_types]
    # 随机打乱类型列表。
    types_sorted = sorted(types)  # 添加排序确保顺序固定
    # 设定随机种子
    np.random.seed(args.seed)
    types_shuffled = types_sorted.copy()  # 避免修改原始列表
    np.random.shuffle(types_shuffled)
    n_types = len(types_shuffled)
    # 计算每个数据集的平均类型数量。
    avg_n_types = n_types // 3
    # 将类型列表分割为训练集、验证集和测试集。
    # train_types = types_shuffled[: avg_n_types + 1]
    # val_types = types_shuffled[ : avg_n_types * 2 + 1]
    # test_types = types_shuffled
    train_types = types_shuffled[: avg_n_types + 1]
    val_types = types_shuffled[avg_n_types + 1 : avg_n_types * 2 + 1]
    test_types = types_shuffled[avg_n_types * 2 + 1 :]


    print("train_types", train_types)
    print("val_types", val_types)
    print("test_types", test_types)

    # 返回分割后的类型列表。
    return train_types, val_types, test_types
# ======================
# 自定义数据集类（适配 Qwen-VL 格式）
# ======================
class QwenVLDataset(Dataset):
    _train_types = None
    _val_types = None
    _test_types = None
    _has_split = False

    def __init__(self, data_paths, image_base_path, processor, mode='train'):
        self.dataset = []
        self.image_base_path = image_base_path
        self.processor = processor
        self.mode = mode

        if not QwenVLDataset._has_split:
            all_data = []
            for data_path in data_paths:
                with open(data_path, 'r', encoding='utf-8') as f:
                    all_data.extend(json.load(f))
            all_types = list(set(item['relation'] for item in all_data))
            train_types, val_types, test_types = split_types(all_types)
            QwenVLDataset._train_types = set(train_types)
            QwenVLDataset._val_types = set(val_types)
            QwenVLDataset._test_types = set(test_types)
            QwenVLDataset._has_split = True

            print(f"✅ Split done with seed={args.seed}:")
            print(f"   Train types ({len(train_types)}): {sorted(train_types)}")
            print(f"   Val types   ({len(val_types)}): {sorted(val_types)}")
            print(f"   Test types  ({len(test_types)}): {sorted(test_types)}")


        for data_path in data_paths:
            with open(data_path, 'r', encoding='utf-8') as f:
                temp_dataset = json.load(f)
                self.dataset.extend(temp_dataset)

        if self.mode == 'train':
            self.dataset = [item for item in self.dataset if item['relation'] in QwenVLDataset._train_types]
            self.available_relations = sorted(relation_map[t] for t in QwenVLDataset._train_types if t in relation_map)
        elif self.mode == 'val':
            self.dataset = [item for item in self.dataset if item['relation'] in QwenVLDataset._val_types]
            self.available_relations = sorted(relation_map[t] for t in QwenVLDataset._val_types if t in relation_map)
        elif self.mode == 'test':
            self.dataset = [item for item in self.dataset if item['relation'] in QwenVLDataset._test_types]
            self.available_relations = sorted(relation_map[t] for t in QwenVLDataset._test_types if t in relation_map)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.options_str = "Options:\n" + "\n".join([
            f"{i + 1}. {rel}" for i, rel in enumerate(self.available_relations)
        ])

        print(f"📊 {self.mode} dataset loaded: {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = os.path.join(self.image_base_path, item["img_id"])
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224,224))        # 获取模型专用的图像令牌
        image_token = processor.image_token

        # 用空格分隔多个令牌，确保被正确解析
        # multiple_image_tokens = f"{image_token} {image_token} {image_token} {image_token}"
        multiple_image_tokens = f"{' '.join([image_token] * 16)}"
        instruction = (
            f"{multiple_image_tokens}\n"
            f"### Task: Identify the relationship between two entities\n"
            f"### Text: {' '.join(item['token'])}\n"
            f"### Head Entity: {item['h']['name']}\n"
            f"### Tail Entity: {item['t']['name']}\n"
            f"### Options: "+ options_str + "\n"
            "Answer only with the relation name, no other text.\n"
            "Relation:"
        )
        # instruction = (
        #     f"{multiple_image_tokens}\n"
        #     f"### Task: Identify the relationship between two entities\n"
        #     f"### Text: {' '.join(item['token'])}\n"
        #     f"### Head Entity: {item['h']['name']}\n"
        #     f"### Tail Entity: {item['t']['name']}\n"
        #     f"{self.options_str}\n"
        #     "Answer ONLY with the EXACT relation name from the options above. "  # 强调必须从选项中选
        #     "Do NOT add any other words, explanations, or punctuation.\n"
        #     "Relation:"
        # )

        target_text = relation_map.get(item["relation"], "none")

        return {
            "text": instruction,  # 包含4个图像令牌
            "image": image,
            "target_text": target_text,
            "id": item.get("id", f"item_{idx}")
        }
# ======================
# 数据整理函数
# ======================
# def collate_fn(batch, processor):
#     texts = [item["text"] for item in batch]
#     images = [item["image"] for item in batch]
#     target_texts = [item["target_text"] for item in batch]
#     ids = [item["id"] for item in batch]
#
#     # 获取图像令牌
#     image_token = processor.image_token
#
#     # # 图像缩放（保持32x32生成4个特征）
#     # resized_images = []
#     # for img in images:
#     #     resized_img = img.resize((32, 32), Image.LANCZOS)
#     #     resized_images.append(resized_img)
#     images = [image.resize((120, 120), Image.LANCZOS) for image in images]
#     # 处理文本和图像，获取模型输入
#     text_inputs = processor.tokenizer(
#         texts,
#         return_tensors="pt",
#         # padding="max_length",
#         padding=True,
#         max_length=2048,
#         truncation=True
#     )
#     image_inputs = processor.image_processor(
#         images,
#         return_tensors="pt"
#     )
#     inputs = {**text_inputs, **image_inputs}
#     input_length = inputs["input_ids"].shape[1]  # 获取输入序列长度（512）
#
#     print("inputs",inputs)
#     print("input_length",input_length)
#     # 关键修改：扩展标签长度以匹配输入序列
#     labels = []
#     for text in target_texts:
#         # 对每个标签进行编码
#         encoded_label = processor.tokenizer(
#             text,
#             padding="max_length",
#             max_length=input_length,  # 标签长度与输入序列一致
#             truncation=True,
#             return_tensors="pt"
#         ).input_ids[0]  # 取单条数据
#
#         # 仅保留目标文本部分，其余位置设为-100（不参与损失计算）
#         # 找到目标文本的结束位置
#         end_idx = (encoded_label == processor.tokenizer.eos_token_id).nonzero().squeeze().item() if (
#                     encoded_label == processor.tokenizer.eos_token_id).any() else len(encoded_label)
#         # 目标文本之后的位置设为-100
#         encoded_label[end_idx + 1:] = -100
#         labels.append(encoded_label)
#
#     # 转换为张量
#     labels = torch.stack(labels, dim=0)
#     # 输入序列中padding的位置，标签也设为-100
#     labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)
#
#     return {
#         "input_ids": inputs["input_ids"],
#         "attention_mask": inputs["attention_mask"],
#         "pixel_values": inputs["pixel_values"],
#         "image_grid_thw": inputs["image_grid_thw"],
#         "labels": labels,
#         "ids": ids,
#         "target_texts": target_texts
#     }
def collate_fn(batch, processor):
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    target_texts = [item["target_text"] for item in batch]
    ids = [item["id"] for item in batch]
    print("texts",texts)
    print("target_texts",target_texts)
    # 获取图像令牌
    image_token = processor.image_token

    # # 图像缩放（保持32x32生成4个特征）
    # resized_images = []
    # for img in images:
    #     resized_img = img.resize((32, 32), Image.LANCZOS)
    #     resized_images.append(resized_img)
    images = [image.resize((120, 120), Image.LANCZOS) for image in images]
    # 处理文本和图像，获取模型输入
    text_inputs = processor.tokenizer(
        texts,
        return_tensors="pt",
        # padding="max_length",
        padding=True,
        max_length=2048,
        truncation=True
    )
    image_inputs = processor.image_processor(
        images,
        return_tensors="pt"
    )
    inputs = {**text_inputs, **image_inputs}
    input_length = inputs["input_ids"].shape[1]  # 获取输入序列长度（512）


    # 处理目标文本
    with processor.tokenizer.as_target_tokenizer():
        labels_encoding = processor.tokenizer(
            text=target_texts,
            padding="max_length",
            max_length=input_length,
            truncation=True,
            return_tensors="pt",
            # add_special_tokens=True  # 确保添加特殊标记
        )
        labels = labels_encoding.input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # print("labels_encoding",labels_encoding)
        print("labels",labels)


    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
        "labels": labels,
        "ids": ids,
        "target_texts": target_texts
    }


# 评估函数
def evaluate(model, test_loader, processor):
    model.eval()
    results = []
    category_correct = {}  # 正确预测的计数
    category_total = {}  # 总预测数的计数
    confusion_matrix = {}  # 真实类别 → 预测结果 → 计数

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            # print("inputs",inputs)
            if batch["pixel_values"] is not None:
                inputs["pixel_values"] = batch["pixel_values"].to(device)

            if "image_grid_thw" in batch and batch["image_grid_thw"] is not None:
                # ✅ 必须传递 image_grid_thw
                inputs["image_grid_thw"] = batch["image_grid_thw"].to(device)
            # 混合精度
            with torch.cuda.amp.autocast():
                # 生成预测
                outputs = model.generate(
                    **inputs,
                    # max_new_tokens=15,    #限制模型最多生成的新 token 数量
                    # num_beams=3,            #即每次保留的候选序列数量
                    # early_stopping=True,    #当模型生成的序列中不再有新的 token 时，停止生成
                    # temperature=0,           #生成时使用的温度参数，值越小，生成的结果越 deterministic
                    # do_sample=False,        #是否使用采样策略，False 表示不使用采样
                    # prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
                )

            # 解码预测
            pred_texts = processor.batch_decode(outputs, skip_special_tokens=True)
            input_texts = processor.batch_decode(batch["input_ids"], skip_special_tokens=True)

            print("pred_texts",pred_texts)
            # print("input_texts",input_texts)

            # 提取预测的关系
            for i, (pred, input_text) in enumerate(zip(pred_texts, input_texts)):
                raw_pred = pred.replace(input_text, "").strip().lower()

                # 尝试匹配关系
                found = False
                for rel in rel_lower:
                    if rel in raw_pred.split() or raw_pred.startswith(rel):
                        prediction = rel_lower[rel]
                        found = True
                        break

                if not found:
                    prediction = "unknown"

                # 获取真实关系
                true_rel = batch["target_texts"][i]  # 在collate_fn中添加target_texts

                if true_rel not in category_total:
                    category_total[true_rel] = 0
                    category_correct[true_rel] = 0
                    confusion_matrix[true_rel] = {}
                category_total[true_rel] += 1

                if prediction == true_rel:
                    category_correct[true_rel] += 1
                else:
                    # ✅ 统计错误预测：真实类别 -> 被预测成什么
                    if prediction not in confusion_matrix[true_rel]:
                        confusion_matrix[true_rel][prediction] = 0
                    confusion_matrix[true_rel][prediction] += 1

                results.append({
                    "id": batch["ids"][i],
                    "prediction": prediction,
                    "true_relation": true_rel,
                    "correct": prediction == true_rel
                })
        category_accuracy = {}
        for category in category_total.keys():
            correct = category_correct.get(category, 0)
            total = category_total[category]
            accuracy = correct / total if total > 0 else 0
            category_accuracy[category] = accuracy
            print(f"Category {category} Accuracy: {accuracy:.4f}")

            # ✅ 打印错误预测统计
        print("\n" + "=" * 50)
        print("❌ 错误预测分析 (Confusion by True Label)")
        print("=" * 50)
        for true_rel, preds in confusion_matrix.items():
            errors = {k: v for k, v in preds.items() if v > 0}
            if errors:
                print(f"\n真实关系: '{true_rel}' 被错分为:")
                # 按频次降序排列
                sorted_errors = sorted(errors.items(), key=lambda x: -x[1])
                for pred_rel, count in sorted_errors:
                    print(f"  → '{pred_rel}': {count} 次")

    return results,category_accuracy

def compute_macro_PRF(predicted_labels, gold_labels, i=-1, empty_label=None):
    '''
    This evaluation function follows work from Sorokin and Gurevych(https://www.aclweb.org/anthology/D17-1188.pdf)
    code borrowed from the following link:
    https://github.com/UKPLab/emnlp2017-relation-extraction/blob/master/relation_extraction/evaluation/metrics.py
    '''
    if i == -1:
        i = len(predicted_labels)

    complete_rel_set = set(gold_labels) - {empty_label}
    avg_prec = 0.0
    avg_rec = 0.0

    for r in complete_rel_set:
        r_indices = [j for j in range(i) if predicted_labels[j] == r]
        tp = sum([predicted_labels[j] == gold_labels[j] for j in r_indices])
        tp_fp = len(r_indices)
        tp_fn = gold_labels.count(r)
        prec = (tp / tp_fp) if tp_fp > 0 else 0
        rec = tp / tp_fn if tp_fn > 0 else 0
        avg_prec += prec
        avg_rec += rec

    f1 = 0
    avg_prec = avg_prec / len(set(predicted_labels[:i]))
    avg_rec = avg_rec / len(complete_rel_set)
    if (avg_rec + avg_prec) > 0:
        f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)

    return avg_prec, avg_rec, f1


# 计算指标函数
def calculate_metrics(results):
    true_labels = [r["true_relation"] for r in results]
    predicted_labels = [r["prediction"] for r in results]

    # 计算整体准确率
    total = len(results)
    correct = sum(1 for r in results if r["prediction"] == r["true_relation"])
    accuracy = correct / total if total > 0 else 0

    # 计算 Precision, Recall, F1 Score
    precision, recall, f1 = compute_macro_PRF(predicted_labels, true_labels)

    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }



# 计算指标函数
def calculate_metrics(results):
    true_labels = [r["true_relation"] for r in results]
    predicted_labels = [r["prediction"] for r in results]

    # 计算整体准确率
    total = len(results)
    correct = sum(1 for r in results if r["prediction"] == r["true_relation"])
    accuracy = correct / total if total > 0 else 0

    # 计算 Precision, Recall, F1 Score
    precision, recall, f1 = compute_macro_PRF(predicted_labels, true_labels)

    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ======================
# 主函数
# ======================
def main():

    model_name = "/home/NCUT/23/jz/Qwen2-VL-7B"

    # 加载 tokenizer 和 processor（需 trust_remote_code）
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    processor = Qwen2VLProcessor.from_pretrained(model_name)
    # tokenizer = processor.tokenizer
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )

    model = model.to(device)
    # 数据路径
    train_path = "/home/NCUT/23/jz/AlignRE_LLaVA/data/new/mnre_llava_train.json"
    val_path = "/home/NCUT/23/jz/AlignRE_LLaVA/data/new/mnre_llava_val.json"
    test_path = "/home/NCUT/23/jz/AlignRE_LLaVA/data/new/mnre_llava_test.json"
    image_base_path = "/home/NCUT/23/jz/AlignRE_LLaVA/data/new/image"
    data_paths = [train_path, val_path, test_path]

    # 构建数据集
    train_dataset = QwenVLDataset(data_paths, image_base_path, processor, 'train')
    val_dataset = QwenVLDataset(data_paths, image_base_path, processor, 'val')
    test_dataset = QwenVLDataset(data_paths, image_base_path, processor, 'test')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor)
    )

    # 设置训练轮数为 5



    test_results, category_accuracy = evaluate(model, test_loader, processor)
    metrics = calculate_metrics(test_results)
    print("seed", args.seed)
    # 注意：evaluate 和结果保存已在 train_with_test_every_epoch 中完成
    print("\n🎉 全部训练与测试已完成！")



if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
    print("seed", args.seed)