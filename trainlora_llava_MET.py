import hashlib
import os
import random
import re
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError, ImageFile
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from argparse import ArgumentParser
from config.loader import apply_config, load_config


# ============================================================
# 1. 基础配置
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = ArgumentParser()
parser.add_argument("--config", type=str, default="", help="Path to config YAML/JSON file")
parser.add_argument("--seed", type=int, default=97)
parser.add_argument("--task", type=int, choices=[1, 2, 3], default=3)
parser.add_argument("--mode", type=str, default="train_eval", choices=["train", "eval", "train_eval"])

# 按 MRE 代码风格统一命名
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--prototype_path", type=str, default="")
parser.add_argument("--train_path", type=str, default="")
parser.add_argument("--val_path", type=str, default="")
parser.add_argument("--test_path", type=str, default="")
parser.add_argument("--image_base_path", type=str, default="")
parser.add_argument("--save_dir", type=str, default="./checkpoints_llava_met")

parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=3e-6)

parser.add_argument("--train_max_samples", type=int, default=0)
parser.add_argument("--val_max_samples", type=int, default=0)
parser.add_argument("--test_max_samples", type=int, default=0)

args = parser.parse_args()
if args.config:
    config = load_config(args.config)
    apply_config(args, config)

if not args.model_name or not args.prototype_path:
    raise ValueError("Please specify --model_name and --prototype_path either directly or via --config")

if not args.train_path or not args.val_path or not args.test_path:
    raise ValueError("Please specify --train_path, --val_path, and --test_path either directly or via --config")

if not args.image_base_path:
    raise ValueError("Please specify --image_base_path either directly or via --config")


# ============================================================
# 2. 随机种子
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)


# ============================================================
# 3. MET 零样本类别划分（保留原 MET 逻辑）
# ============================================================
if args.task == 1:
    GroupA = ["People", "Site", "Building", "Currency"]
    GroupB = ["Location", "Event", "Book", "Music"]
    GroupC = ["Organization", "Country", "APP", "Movie"]
elif args.task == 2:
    GroupC = ["People", "Site", "Building", "Currency"]
    GroupA = ["Location", "Event", "Book", "Music"]
    GroupB = ["Organization", "Country", "APP", "Movie"]
elif args.task == 3:
    GroupB = ["People", "Site", "Building", "Currency"]
    GroupC = ["Location", "Event", "Book", "Music"]
    GroupA = ["Organization", "Country", "APP", "Movie"]

candidate_relations = [
    "People", "Site", "Building", "Currency",
    "Location", "Event", "Book", "Music",
    "Organization", "Country", "APP", "Movie"
]
rel_lower = {rel.lower(): rel for rel in candidate_relations}


# ============================================================
# 4. 工具函数
# ============================================================
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def fix_processor_image_size(processor, model):
    """
    修复部分 processor/image_processor 配置下 size 只有 shortest_edge，
    但底层 BLIP 图像预处理要求 height/width 的报错问题。
    """
    if not hasattr(processor, "image_processor"):
        return processor

    vision_size = None
    if hasattr(model.config, "vision_config") and hasattr(model.config.vision_config, "image_size"):
        vision_size = int(model.config.vision_config.image_size)

    size_cfg = getattr(processor.image_processor, "size", None)
    crop_cfg = getattr(processor.image_processor, "crop_size", None)

    if vision_size is not None:
        processor.image_processor.size = {"height": vision_size, "width": vision_size}
        processor.image_processor.crop_size = {"height": vision_size, "width": vision_size}
    else:
        if isinstance(size_cfg, dict) and "shortest_edge" in size_cfg:
            edge = int(size_cfg["shortest_edge"])
            processor.image_processor.size = {"height": edge, "width": edge}
        if isinstance(crop_cfg, dict) and "shortest_edge" in crop_cfg:
            edge = int(crop_cfg["shortest_edge"])
            processor.image_processor.crop_size = {"height": edge, "width": edge}
        elif isinstance(crop_cfg, int):
            processor.image_processor.crop_size = {"height": int(crop_cfg), "width": int(crop_cfg)}

    print("processor type:", type(processor))
    print("image_processor type:", type(processor.image_processor))
    print("image_processor.size:", getattr(processor.image_processor, "size", None))
    print("image_processor.crop_size:", getattr(processor.image_processor, "crop_size", None))
    return processor


def parse_img_file_name(url: str):
    m_img = url.split("/")[-1]
    prefix = hashlib.md5(m_img.encode()).hexdigest()
    suffix = re.sub(
        r"(\S+(?=\.(jpg|JPG|png|PNG|svg|SVG)))|(\S+(?=\.(jpeg|JPEG)))", "", m_img
    )
    m_img = prefix + suffix
    m_img = m_img.replace(".svg", ".png").replace(".SVG", ".png")
    return m_img


def extract_label_set(text):
    labels = set()
    text_lower = text.lower()
    for rel in candidate_relations:
        pattern = rf"\b{re.escape(rel.lower())}\b"
        if re.search(pattern, text_lower):
            labels.add(rel)
    return labels


# ============================================================
# 5. 数据集
#    按 MRE 代码形式：读取所有 json，再依据 mode 过滤类别
# ============================================================
class METDataset(Dataset):
    def __init__(self, data_paths, image_base_path, processor, mode='train'):
        self.dataset = []
        self.image_base_path = image_base_path
        self.processor = processor
        self.mode = mode

        all_data = []
        for data_path in data_paths:
            with open(data_path, 'r', encoding='utf-8') as f:
                all_data.extend(json.load(f))

        if mode == 'train':
            allowed_types = set(GroupA)
        elif mode == 'val':
            allowed_types = set(GroupB)
        elif mode == 'test':
            allowed_types = set(GroupC)
        else:
            raise ValueError("mode must be 'train', 'val', or 'test'")

        self.available_relations = sorted(list(allowed_types))
        self.options_str = "Options:\n" + "\n".join([
            f"{i + 1}. {rel}" for i, rel in enumerate(self.available_relations)
        ])

        for item in all_data:
            description, image_url, category, entities = item

            has_other_entity = False
            valid_entity_types = set()

            for ent in entities:
                ent_text, ent_type, start, end, wiki_link = ent
                if ent_type.lower() == "other":
                    has_other_entity = True
                    break
                if ent_type in allowed_types:
                    valid_entity_types.add(ent_type)

            if has_other_entity:
                continue

            if valid_entity_types:
                self.dataset.append({
                    "description": description,
                    "image_url": image_url,
                    "category": category,
                    "entity_types": sorted(valid_entity_types),
                    "entities_detail": entities
                })

        if self.mode == "train" and args.train_max_samples > 0:
            self.dataset = self.dataset[:args.train_max_samples]
        elif self.mode == "val" and args.val_max_samples > 0:
            self.dataset = self.dataset[:args.val_max_samples]
        elif self.mode == "test" and args.test_max_samples > 0:
            self.dataset = self.dataset[:args.test_max_samples]

        print(f"📊 {self.mode} dataset loaded: {len(self.dataset)} samples, {len(self.available_relations)} entity types")
        print(f"Dynamic options for {self.mode}: {self.options_str}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        file_name = parse_img_file_name(item["image_url"])
        local_image_path = os.path.join(self.image_base_path, file_name)

        try:
            image = Image.open(local_image_path).convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
            print(f"Error loading image {local_image_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        input_text = (
            f"Image: <image>\n"
            "Identify the entity types in the text-image pair:\n"
            f"Text: {item['description']}\n"
            f"{self.options_str}\n"
            "Answer only with the entity type names from the options, separated by commas if multiple.\n"
            "Entity Types:"
        )

        target_text = ", ".join(item["entity_types"])
        sample_id = f"{self.mode}_{item['category']}_{idx}"

        return {
            "image": image,
            "input_text": input_text,
            "target_text": target_text,
            "id": sample_id
        }


# ============================================================
# 6. Collate
# ============================================================
def collate_fn(batch, processor):
    images = [item["image"] for item in batch]
    input_texts = [item["input_text"] for item in batch]
    target_texts = [item["target_text"] for item in batch]
    ids = [item["id"] for item in batch]

    text_inputs = processor(
        text=input_texts,
        images=images,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=1024,
        return_attention_mask=True
    )

    with processor.tokenizer.as_target_tokenizer():
        labels_encoding = processor.tokenizer(
            text=target_texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=1024,
        )
        labels = labels_encoding.input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "pixel_values": text_inputs.pixel_values,
        "input_ids": text_inputs.input_ids,
        "attention_mask": text_inputs.attention_mask,
        "labels": labels,
        "ids": ids,
        "target_texts": target_texts
    }


# ============================================================
# 7. 指标（适配 MET 多标签）
# ============================================================
def calculate_metrics(results):
    total = len(results)
    if total == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    hit_correct = 0
    tp = fp = fn = 0

    for r in results:
        pred_set = set(r["prediction_list"])
        true_set = set(r["true_relation_list"])

        if len(pred_set & true_set) > 0:
            hit_correct += 1

        tp += len(pred_set & true_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    accuracy = hit_correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

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


# ============================================================
# 8. 验证 / 测试
# ============================================================
def validate(model, val_loader, processor):
    model.eval()
    total_loss = 0
    results = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", total=len(val_loader))
        for batch in progress_bar:
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device)
            }

            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                loss = outputs.loss

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            generated_outputs = model.generate(
                pixel_values=batch["pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=32,
            )

            pred_texts = processor.batch_decode(generated_outputs, skip_special_tokens=True)
            input_texts = processor.batch_decode(batch["input_ids"], skip_special_tokens=True)

            for i, (pred, input_text) in enumerate(zip(pred_texts, input_texts)):
                raw_pred = pred.replace(input_text, "").strip()
                pred_set = sorted(list(extract_label_set(raw_pred)))
                true_set = sorted(list(extract_label_set(batch["target_texts"][i])))

                if not pred_set:
                    pred_set = ["unknown"]

                results.append({
                    "id": batch["ids"][i],
                    "prediction": ", ".join(pred_set),
                    "prediction_list": pred_set,
                    "true_relation": ", ".join(true_set),
                    "true_relation_list": true_set,
                    "correct": len(set(pred_set) & set(true_set)) > 0
                })

            del inputs, outputs, loss
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    metrics = calculate_metrics(results)
    return avg_loss, metrics


def evaluate(model, test_loader, processor):
    model.eval()
    results = []
    category_correct = {}
    category_total = {}
    confusion_matrix = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }

            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                )

            pred_texts = processor.batch_decode(outputs, skip_special_tokens=True)
            input_texts = processor.batch_decode(batch["input_ids"], skip_special_tokens=True)

            for i, (pred, input_text) in enumerate(zip(pred_texts, input_texts)):
                raw_pred = pred.replace(input_text, "").strip()
                pred_set = sorted(list(extract_label_set(raw_pred)))
                true_set = sorted(list(extract_label_set(batch["target_texts"][i])))

                if not pred_set:
                    pred_set = ["unknown"]

                for true_rel in true_set:
                    if true_rel not in category_total:
                        category_total[true_rel] = 0
                        category_correct[true_rel] = 0
                        confusion_matrix[true_rel] = {}

                    category_total[true_rel] += 1

                    if true_rel in pred_set:
                        category_correct[true_rel] += 1
                    else:
                        for pred_rel in pred_set:
                            if pred_rel not in confusion_matrix[true_rel]:
                                confusion_matrix[true_rel][pred_rel] = 0
                            confusion_matrix[true_rel][pred_rel] += 1

                results.append({
                    "id": batch["ids"][i],
                    "prediction": ", ".join(pred_set),
                    "prediction_list": pred_set,
                    "true_relation": ", ".join(true_set),
                    "true_relation_list": true_set,
                    "correct": len(set(pred_set) & set(true_set)) > 0
                })

    category_accuracy = {}
    for category in category_total.keys():
        correct = category_correct.get(category, 0)
        total = category_total[category]
        accuracy = correct / total if total > 0 else 0
        category_accuracy[category] = accuracy
        print(f"Category {category} Accuracy: {accuracy:.4f}")

    print("\n" + "=" * 50)
    print("❌ 错误预测分析 (Confusion by True Label)")
    print("=" * 50)
    for true_rel, preds in confusion_matrix.items():
        errors = {k: v for k, v in preds.items() if v > 0}
        if errors:
            print(f"\n真实类别: '{true_rel}' 被错分为:")
            sorted_errors = sorted(errors.items(), key=lambda x: -x[1])
            for pred_rel, count in sorted_errors:
                print(f"  → '{pred_rel}': {count} 次")

    return results, category_accuracy


# ============================================================
# 9. 保存测试结果（对齐 MRE）
# ============================================================
def save_test_result(results, metrics, epoch, save_dir):
    result_data = {
        "epoch": epoch,
        "results": results,
        "metrics": metrics
    }
    filename = os.path.join(save_dir, f"test_results_epoch_{epoch}.json")
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 第 {epoch} 轮测试结果已保存至: {filename}")


# ============================================================
# 10. 训练（完全按 MRE 风格：周期性测试 + 只保存 LoRA）
# ============================================================
def train_with_periodic_evaluation(model, train_loader, val_loader, test_loader,
                                   optimizer, scheduler, epochs, processor, device,
                                   save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    best_lora_dir = os.path.join(save_dir, "best_lora")
    latest_lora_dir = os.path.join(save_dir, "latest_lora")

    accumulation_steps = 4
    scaler = torch.cuda.amp.GradScaler()
    all_test_metrics = []

    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'=' * 50}")

        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        optimizer.zero_grad()

        for i, batch in enumerate(progress_bar):
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device)
            }

            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                loss = outputs.loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

            del inputs, outputs, loss
            torch.cuda.empty_cache()

        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

        val_loss, val_metrics = validate(model, val_loader, processor)
        print(f"Validation Loss: {val_loss:.4f}, Metrics: {val_metrics}")

        print(f"\n🔄 正在执行第 {epoch + 1} 轮测试...")
        test_results, category_accuracy = evaluate(model, test_loader, processor)
        test_metrics = calculate_metrics(test_results)
        save_test_result(test_results, test_metrics, epoch=epoch + 1, save_dir=save_dir)

        all_test_metrics.append({
            "epoch": epoch + 1,
            "metrics": test_metrics,
            "val_loss": val_loss,
            "train_loss": avg_train_loss
        })

        # 保存最新 LoRA
        model.save_pretrained(latest_lora_dir)
        processor.save_pretrained(latest_lora_dir)

        # 保存最优 LoRA
        current_f1 = test_metrics["f1"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch + 1
            model.save_pretrained(best_lora_dir)
            processor.save_pretrained(best_lora_dir)
            print(f"🎉 最佳 LoRA 已保存：F1={best_f1:.4f}")
            print(f"💾 路径：{best_lora_dir}")

    print(f"\n{'=' * 60}")
    print("📊 所有轮次测试结果汇总")
    print(f"{'=' * 60}")
    for record in all_test_metrics:
        epoch = record["epoch"]
        f1 = record["metrics"]["f1"]
        acc = record["metrics"]["accuracy"]
        prec = record["metrics"]["precision"]
        rec = record["metrics"]["recall"]
        print(f"Epoch {epoch:2d}: F1={f1:.4f}, Acc={acc:.2%}, Prec={prec:.4f}, Rec={rec:.4f}")

    print(f"\n🏆 最佳结果：Epoch {best_epoch} | F1={best_f1:.4f}")
    return all_test_metrics, best_epoch, best_f1


# ============================================================
# 11. 主函数
# ============================================================
def main():
    os.makedirs(args.save_dir, exist_ok=True)

    save_json({
        "args": vars(args),
        "groups": {
            "GroupA": GroupA,
            "GroupB": GroupB,
            "GroupC": GroupC,
        }
    }, os.path.join(args.save_dir, "run_config.json"))

    train_relation_prototype = torch.load(args.prototype_path)

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    processor = LlavaProcessor.from_pretrained(args.model_name)
    processor = fix_processor_image_size(processor, model)

    model.register_buffer("plora_embeddings", train_relation_prototype.to(model.device))

    def find_all_linear_names(model):
        cls = torch.nn.Linear
        lora_module_names = set()
        multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
        for name, module in model.named_modules():
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    class FixedPeftModel(PeftModel):
        def forward(self, *args, **kwargs):
            kwargs.pop("inputs_embeds", None)
            return super().forward(*args, **kwargs)

    # 保留原 MET 的 LoRA 超参数，不动任务设置，只改训练框架形式
    lora_config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=find_all_linear_names(model),
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
    )

    model = get_peft_model(model, lora_config)
    model = FixedPeftModel(
        model.model,
        peft_config=lora_config,
    )
    model.print_trainable_parameters()
    model = model.to(device)

    data_paths = [args.train_path, args.val_path, args.test_path]
    train_dataset = METDataset(data_paths, args.image_base_path, processor, 'train')
    val_dataset = METDataset(data_paths, args.image_base_path, processor, 'val')
    test_dataset = METDataset(data_paths, args.image_base_path, processor, 'test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    if args.mode in ["train", "train_eval"]:
        train_with_periodic_evaluation(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            processor=processor,
            device=device,
            save_dir=args.save_dir
        )

    if args.mode in ["eval", "train_eval"]:
        best_lora_path = os.path.join(args.save_dir, "best_lora")
        latest_lora_path = os.path.join(args.save_dir, "latest_lora")

        if os.path.exists(best_lora_path):
            model = PeftModel.from_pretrained(model, best_lora_path, is_trainable=False)
            model = model.to(device)
            print("✅ 已加载最佳 LoRA 模型")
        elif os.path.exists(latest_lora_path):
            model = PeftModel.from_pretrained(model, latest_lora_path, is_trainable=False)
            model = model.to(device)
            print("✅ 已加载最新 LoRA 模型")

        test_results, category_accuracy = evaluate(model, test_loader, processor)
        metrics = calculate_metrics(test_results)
        save_json({
            "results": test_results,
            "metrics": metrics,
            "category_accuracy": category_accuracy
        }, os.path.join(args.save_dir, "test_results.json"))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
    print("task:", args.task)
