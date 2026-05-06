import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, LlavaForConditionalGeneration, \
    CLIPImageProcessor, LlavaProcessor, Blip2Processor, Blip2ForConditionalGeneration, get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import re
import os
from sklearn.metrics import precision_recall_fscore_support
from typing import List, Optional, Callable
from peft import LoraConfig, get_peft_model, PeftModel
from zmq.utils.garbage import gc

# 设置设备
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = "cuda" if torch.cuda.is_available() else "cpu"
# 初始化模型
model_name = "/home/NCUT/23/jz/NewUMIE/MM-InstructEval-main/blip2-flan-t5-xxl"
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True  # 使用8位量化减少内存
)
processor = Blip2Processor.from_pretrained(model_name)
tokenizer = processor.tokenizer
image_processor = processor.image_processor
# 关系映射表
relation_map = {
    "P19": "place_of_birth",
    "P22": "parent",
    "P26": "couple",
    "P27": "nationality",
    "P31": "member_of",
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
    "P259": "alternate_names",
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
    "member_of",
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
    "alternate_names",
    "race",
    "siblings",
    "none"
]

rel_lower = {rel.lower(): rel for rel in candidate_relations}  # 标准化映射

# 构建带编号的选项字符串（固定格式）
options_str = "Options:\n" + "\n".join([f"{i + 1}. {rel}" for i, rel in enumerate(candidate_relations)])


# 自定义数据集类
class MNREDataset(Dataset):
    def __init__(self, data_path, image_base_path, processor, mode='train'):
        with open(data_path) as f:
            self.dataset = json.load(f)
        self.image_base_path = image_base_path
        self.processor = processor
        self.mode = mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 加载图像
        image_path = os.path.join(self.image_base_path, item["img_id"])
        image = Image.open(image_path).convert("RGB")

        image = image_processor(image, return_tensors="pt")
        # 构建输入文本
        input_text = (
            "Identify the relationship between entities:\n"
            f"Text: {' '.join(item['token'])}\n"
            f"Head Entity: {item['h']['name']} (Position: {item['h']['pos']})\n"
            f"Tail Entity: {item['t']['name']} (Position: {item['t']['pos']})\n"
            f"{options_str}\n"
            "Answer only with the relation name, no other text.\n"
            "Relation:"
        )

        # 目标文本（关系名称）
        target_text = relation_map.get(item["relation"])

        return {
            "image": image,
            "input_text": input_text,
            "target_text": target_text,
            "id": item.get("id", f"item_{idx}")
        }


# 数据整理函数
def collate_fn(batch, processor):
    images = [item["image"] for item in batch]
    input_texts = [item["input_text"] for item in batch]
    target_texts = [item["target_text"] for item in batch]
    ids = [item["id"] for item in batch]
    print("target_texts",target_texts)
    # 处理图像
    # image_inputs = processor(images=images, return_tensors="pt")
    # image_min = image_tensor.min()
    # image_max = image_tensor.max()
    # image_inputs = (image_tensor - image_min) / (image_max - image_min)
    # 处理文本输入
    text_inputs = processor(
        text=input_texts,
        images=images,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )


    # 处理目标文本
    with processor.tokenizer.as_target_tokenizer():
        labels_encoding = processor(
            target_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True  # 确保添加特殊标记
        )
        labels = labels_encoding.input_ids


    return {
        "input_ids": text_inputs.input_ids,
        "attention_mask": text_inputs.attention_mask,
        "pixel_values": text_inputs.pixel_values,
        "labels": labels,
        "ids": ids,
        "target_texts": target_texts  # 保留原始文本用于评估
    }



# 训练函数
def train(model, train_loader, val_loader, optimizer, scheduler, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            # 将数据移动到设备
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device)
            }


            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(**inputs)
            loss = outputs.loss

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 参数更新
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # 释放内存
            del inputs, outputs, loss
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        # 每个epoch结束后验证
        val_loss = validate(model, val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

    #     # 保存模型检查点
    #     checkpoint_path = f"model_epoch_{epoch + 1}.bin"
    #     torch.save(model.state_dict(), checkpoint_path)
    #     print(f"Saved checkpoint to {checkpoint_path}")
    #
    # # 训练完成后保存最终模型
    # final_model_path = "final_model.bin"
    # torch.save(model.state_dict(), final_model_path)
    # print(f"Saved final model to {final_model_path}")

        if isinstance(model.model, PeftModel):
            model.model.save_pretrained(f"lora_epoch_{epoch + 1}")
        else:
            print("Warning: Model is not a PeftModel instance")

# 验证函数
def validate(model, val_loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device)
            }

            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()

            # 释放内存
            del inputs, outputs, loss
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


# 评估函数
def evaluate(model, test_loader, processor):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }

            # 生成预测
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                num_beams=3,
                early_stopping=True,
                temperature=0.8,
                do_sample=True
            )

            # 解码预测
            pred_texts = processor.batch_decode(outputs, skip_special_tokens=True)
            input_texts = processor.batch_decode(batch["input_ids"], skip_special_tokens=True)

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

                results.append({
                    "id": batch["ids"][i],
                    "prediction": prediction,
                    "true_relation": true_rel,
                    "correct": prediction == true_rel
                })

    return results

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


def main():
    # 初始化模型
    model_name = "/home/NCUT/23/jz/NewUMIE/MM-InstructEval-main/blip2-flan-t5-xxl"
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit = True  # 使用8位量化减少内存
    )
    processor = Blip2Processor.from_pretrained(model_name)

    # 冻结视觉和QFormer部分
    for name, param in model.named_parameters():
        if "vision_model" in name or "qformer" in name:
            param.requires_grad = False

    class FixedPeftModel(PeftModel):

        def forward(self, *args, **kwargs):
                # 移除可能导致冲突的非法参数
                # 过滤非法参数（如 inputs_embeds）
                kwargs.pop("inputs_embeds", None)
                return super().forward(*args, **kwargs)
    # 添加LoRA适配器
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",  # Query 投影
            "v_proj",  # Value 投影
            "o_proj",  # 输出投影
            "wi",  # 前馈网络第一层
            "wo",  # 前馈网络输出层
        ],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",  # 使用高斯初始化
    )

    model = get_peft_model(model, lora_config)
    model = FixedPeftModel(
        model.model,  # 基础模型
        peft_config=lora_config,  # LoRA配置
    )
    model.print_trainable_parameters()

    # 数据集路径
    train_path = "/home/NCUT/23/jz/AlignRE_LLaVA/data/new/test.json"
    val_path = "/home/NCUT/23/jz/AlignRE_LLaVA/data/new/test.json"
    test_path = "/home/NCUT/23/jz/AlignRE_LLaVA/data/new/test.json"
    image_base_path = "/home/NCUT/23/jz/AlignRE_LLaVA/data/new/image"

    # 创建数据集
    train_dataset = MNREDataset(train_path, os.path.join(image_base_path), processor, 'train')
    val_dataset = MNREDataset(val_path, os.path.join(image_base_path), processor, 'val')
    test_dataset = MNREDataset(test_path, os.path.join(image_base_path), processor, 'test')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
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

    # 设置优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.01)
    total_steps = len(train_loader) * 3  # 减少epoch数
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # 训练模型
    train(model, train_loader, val_loader, optimizer, scheduler, epochs=3)

    # 测试模型
    test_results = evaluate(model, test_loader, processor)
    metrics = calculate_metrics(test_results)

    # 保存测试结果
    with open("test_results.json", "w") as f:
        json.dump({
            "results": test_results,
            "metrics": metrics
        }, f, indent=2)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
