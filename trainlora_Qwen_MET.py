import hashlib
import random
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from transformers import (
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import json
import os
from peft import LoraConfig, get_peft_model, PeftModel
from argparse import ArgumentParser
from config.loader import apply_config, load_config

# 统一实验设置
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument("--config", type=str, default="", help="Path to config YAML/JSON file")
parser.add_argument("--seed", type=int, default=97)  # 16,67,97
parser.add_argument("--task", type=int, choices=[1, 2, 3], default=1)

# ===== 你要求补上的参数化入口 =====
parser.add_argument("--mode", type=str, default="train_eval", choices=["train", "eval", "train_eval"])
parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--prototype_path", type=str, default="")
parser.add_argument("--train_path", type=str, default="")
parser.add_argument("--val_path", type=str, default="")
parser.add_argument("--test_path", type=str, default="")
parser.add_argument("--image_dir", type=str, default="")
parser.add_argument("--output_dir", type=str, default="./checkpoints_qwen_met")

# ===== 保持原实验设置，同时补出可调入口 =====
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=2e-5)

# ===== 少样本验机 =====
parser.add_argument("--train_max_samples", type=int, default=0)
parser.add_argument("--val_max_samples", type=int, default=0)
parser.add_argument("--test_max_samples", type=int, default=0)

args = parser.parse_args()
if args.config:
    config = load_config(args.config)
    apply_config(args, config)

if not args.model_path or not args.prototype_path:
    raise ValueError("Please specify --model_path and --prototype_path either directly or via --config")

if not args.train_path or not args.val_path or not args.test_path:
    raise ValueError("Please specify --train_path, --val_path, and --test_path either directly or via --config")

if not args.image_dir:
    raise ValueError("Please specify --image_dir either directly or via --config")


def set_seed(seed: int):
    """设置随机数种子，确保代码在每次运行时产生的随机结果具有可重复性。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)
task = args.task

if task == 1:
    GroupA = ["People", "Site", "Building", "Currency"]
    GroupB = ["Location", "Event", "Book", "Music"]
    GroupC = ["Organization", "Country", "APP", "Movie"]
elif task == 2:
    GroupC = ["People", "Site", "Building", "Currency"]
    GroupA = ["Location", "Event", "Book", "Music"]
    GroupB = ["Organization", "Country", "APP", "Movie"]
elif task == 3:
    GroupB = ["People", "Site", "Building", "Currency"]
    GroupC = ["Location", "Event", "Book", "Music"]
    GroupA = ["Organization", "Country", "APP", "Movie"]

candidate_relations = [
    "People", "Site", "Building", "Currency", "Location", "Event", "Book", "Music",
    "Organization", "Country", "APP", "Movie"
]
rel_lower = {rel.lower(): rel for rel in candidate_relations}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class METDataset(Dataset):
    def __init__(self, data_paths, image_base_path, processor, mode='train'):
        self.dataset = []
        self.image_base_path = image_base_path
        self.processor = processor
        self.mode = mode

        if mode == 'train':
            allowed_types = set(GroupA)
        elif mode == 'val':
            allowed_types = set(GroupB)
        elif mode == 'test':
            allowed_types = set(GroupC)
        else:
            raise ValueError("mode must be 'train', 'val', or 'test'")

        self.dataset = self._load_data(data_paths, allowed_types)

        # ===== 最小新增：少样本截断 =====
        if self.mode == "train" and args.train_max_samples > 0:
            self.dataset = self.dataset[:args.train_max_samples]
        elif self.mode == "val" and args.val_max_samples > 0:
            self.dataset = self.dataset[:args.val_max_samples]
        elif self.mode == "test" and args.test_max_samples > 0:
            self.dataset = self.dataset[:args.test_max_samples]

    def _load_data(self, data_paths, allowed_types):
        """加载并过滤数据"""
        for data_path in data_paths:
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                dataset = []
                for item in raw_data:
                    description, image_url, category, entities = item
                    has_other_entity = False

                    for ent in entities:
                        ent_text, ent_type, start, end, wiki_link = ent
                        if ent_type.lower() == "other":
                            has_other_entity = True
                            break
                    if has_other_entity:
                        continue

                    valid_entity_types = set()
                    for ent in entities:
                        ent_text, ent_type, start, end, wiki_link = ent
                        if ent_type in allowed_types:
                            valid_entity_types.add(ent_type)

                    if valid_entity_types:
                        dataset.append({
                            "description": description,
                            "image_url": image_url,
                            "category": category,
                            "entity_types": sorted(valid_entity_types),
                            "entities_detail": entities
                        })

                # 保持你原代码行为，不改动逻辑
                return dataset

    def __len__(self):
        return len(self.dataset)

    def get_group_options(self):
        if self.mode == 'train':
            return GroupA
        elif self.mode == 'val':
            return GroupB
        elif self.mode == 'test':
            return GroupC
        else:
            raise ValueError("mode must be 'train', 'val', or 'test'")

    def __parse_img_file_name(self, url: str):
        m_img = url.split("/")[-1]
        prefix = hashlib.md5(m_img.encode()).hexdigest()
        suffix = re.sub(
            r"(\S+(?=\.(jpg|JPG|png|PNG|svg|SVG)))|(\S+(?=\.(jpeg|JPEG)))", "", m_img
        )
        m_img = prefix + suffix
        m_img = m_img.replace(".svg", ".png").replace(".SVG", ".png")
        return m_img

    def __getitem__(self, idx):
        item = self.dataset[idx]

        description = item["description"]
        image_url = item["image_url"]
        category = item["category"]
        entity_types = item["entity_types"]

        print("entity_types是", entity_types)
        file_name = self.__parse_img_file_name(image_url)
        local_image_path = os.path.join(self.image_base_path, file_name)

        try:
            from PIL import ImageFile
            image = Image.open(local_image_path).convert("RGB")
            image = image.resize((224, 224))
            Image.MAX_IMAGE_PIXELS = None
            ImageFile.LOAD_TRUNCATED_IMAGES = True
        except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
            print(f"Error loading image {local_image_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        current_options = self.get_group_options()
        dynamic_options_str = "Options:\n" + "\n".join([f"{rel}" for i, rel in enumerate(current_options)])

        print(f"📊 {self.mode} dataset loaded: {len(self.dataset)} samples, {len(entity_types)} entity types")
        print(f"Dynamic options for {self.mode}: {dynamic_options_str}")

        # 保持你原来的 Qwen 图像 token 逻辑
        image_token = self.processor.image_token
        multiple_image_tokens = f"{' '.join([image_token] * 16)}"

        input_text = (
            f"{multiple_image_tokens}\n"
            "### Task: Identify the entities\n"
            f"### Text: {description} \n"
            f"{dynamic_options_str}\n"
            "### Instruction: Output ONLY the entities name from the options\n"
            "### Answer:"
        )

        target_text = ", ".join(sorted(entity_types)) if entity_types else "None"
        sample_id = f"{self.mode}_{category}_{idx}"

        return {
            "image": image,
            "input_text": input_text,
            "target_text": target_text,
            "id": sample_id
        }


def collate_fn(batch, processor):
    images = [item["image"] for item in batch]
    input_texts = [item["input_text"] for item in batch]
    target_texts = [item["target_text"] for item in batch]

    # 保持你原来的 Qwen 图像缩放和 image_grid_thw 逻辑
    images = [image.resize((120, 120), Image.LANCZOS) for image in images]

    text_inputs = processor.tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        max_length=2048,
        truncation=True
    )
    image_inputs = processor.image_processor(
        images,
        return_tensors="pt"
    )

    inputs = {**text_inputs, **image_inputs}
    input_length = inputs["input_ids"].shape[1]

    with processor.tokenizer.as_target_tokenizer():
        labels_encoding = processor.tokenizer(
            text=target_texts,
            padding="max_length",
            max_length=input_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = labels_encoding.input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
        "labels": labels,
        "target_text": target_texts
    }


def compute_macro_PRF(predicted_labels, gold_labels, i=-1, empty_label=None):
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

    pred_set = set(predicted_labels[:i])
    if len(pred_set) == 0 or len(complete_rel_set) == 0:
        return 0.0, 0.0, 0.0

    f1 = 0
    avg_prec = avg_prec / len(pred_set)
    avg_rec = avg_rec / len(complete_rel_set)
    if (avg_rec + avg_prec) > 0:
        f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)

    return avg_prec, avg_rec, f1


def calculate_metrics(results):
    true_labels = [r["true_relation"] for r in results]
    predicted_labels = [r["prediction"] for r in results]

    total = len(results)
    correct = sum(1 for r in results if r["prediction"] == r["true_relation"])
    accuracy = correct / total if total > 0 else 0

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


def validate(model, val_loader, processor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0
    results = []

    if task == 1:
        GroupB = ["Location", "Event", "Book", "Music"]
    elif task == 2:
        GroupB = ["Organization", "Country", "APP", "Movie"]
    elif task == 3:
        GroupB = ["People", "Site", "Building", "Currency"]

    valid_categories = set(GroupB)

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", total=len(val_loader))
        for batch in progress_bar:
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
                "image_grid_thw": batch["image_grid_thw"].to(device)
            }

            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            generated_outputs = model.generate(
                pixel_values=batch["pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                image_grid_thw=batch["image_grid_thw"].to(device),
            )

            pred_texts = processor.batch_decode(generated_outputs, skip_special_tokens=True)
            input_texts = processor.batch_decode(batch["input_ids"], skip_special_tokens=True)
            print("pred_texts", pred_texts)

            for i, (pred, input_text) in enumerate(zip(pred_texts, input_texts)):
                raw_pred = pred.replace(input_text, "").strip().lower()

                found = False
                for rel in rel_lower:
                    if rel in raw_pred.split() or raw_pred.startswith(rel):
                        prediction = rel_lower[rel]
                        found = True
                        break
                if not found:
                    prediction = "unknown"

                true_rel_full = batch["target_text"][i]
                true_categories = [cat.strip() for cat in true_rel_full.split(',')]
                valid_true_categories = [cat for cat in true_categories if cat in valid_categories]
                if not valid_true_categories:
                    continue

                valid_true_categories_str = ", ".join(valid_true_categories) if valid_true_categories else "None"
                results.append({
                    "prediction": prediction,
                    "true_relation": valid_true_categories_str,
                    "correct": prediction in valid_true_categories
                })

            del inputs, outputs, loss
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(val_loader)
    metrics = calculate_metrics(results)
    return avg_loss, metrics, results


def evaluate(model, test_loader, processor):
    model.eval()
    results = []
    category_correct = {}
    category_total = {}
    confusion_matrix = {}

    if task == 1:
        GroupC = ["Organization", "Country", "APP", "Movie"]
    elif task == 2:
        GroupC = ["People", "Site", "Building", "Currency"]
    elif task == 3:
        GroupC = ["Location", "Event", "Book", "Music"]

    valid_categories = set(GroupC)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image_grid_thw": batch["image_grid_thw"].to(device)
            }

            outputs = model.generate(**inputs)

            pred_texts = processor.batch_decode(outputs, skip_special_tokens=True)
            input_texts = processor.batch_decode(batch["input_ids"], skip_special_tokens=True)
            print("pred_texts", pred_texts)

            for i, (pred, input_text) in enumerate(zip(pred_texts, input_texts)):
                raw_pred = pred.replace(input_text, "").strip().lower()

                found = False
                for rel in rel_lower:
                    if rel in raw_pred.split() or raw_pred.startswith(rel):
                        prediction = rel_lower[rel]
                        found = True
                        break
                if not found:
                    prediction = "unknown"

                true_rel_full = batch["target_text"][i]
                true_categories = [cat.strip() for cat in true_rel_full.split(',')]
                valid_true_categories = [cat for cat in true_categories if cat in valid_categories]

                for true_rel in valid_true_categories:
                    if true_rel not in category_total:
                        category_total[true_rel] = 0
                        category_correct[true_rel] = 0
                        confusion_matrix[true_rel] = {}
                    category_total[true_rel] += 1

                    if prediction in valid_true_categories:
                        category_correct[true_rel] += 1
                    else:
                        if prediction not in confusion_matrix[true_rel]:
                            confusion_matrix[true_rel][prediction] = 0
                        confusion_matrix[true_rel][prediction] += 1

                valid_true_categories_str = ", ".join(valid_true_categories) if valid_true_categories else "None"
                results.append({
                    "prediction": prediction,
                    "true_relation": valid_true_categories_str,
                    "correct": prediction in valid_true_categories
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
                print(f"\n真实关系: '{true_rel}' 被错分为:")
                sorted_errors = sorted(errors.items(), key=lambda x: -x[1])
                for pred_rel, count in sorted_errors:
                    print(f"  → '{pred_rel}': {count} 次")

    return results, category_accuracy


def train(model, train_loader, val_loader, optimizer, scheduler, epochs, processor, device, output_dir="checkpoints"):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # 只保存 LoRA 适配器路径
    latest_adapter_dir = os.path.join(output_dir, "latest_adapter")
    best_adapter_dir = os.path.join(output_dir, "best_adapter")

    # 训练状态（仅优化器、调度器、epoch，不包含模型权重）
    latest_training_state_path = os.path.join(output_dir, "latest_training_state.pth")
    best_training_state_path = os.path.join(output_dir, "best_training_state.pth")

    model.train()
    best_f1 = -1.0
    history = []

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()

        for batch in progress_bar:
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
                "image_grid_thw": batch["image_grid_thw"].to(device)
            }

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")

            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            del inputs, outputs, loss
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        val_loss, metrics, val_results = validate(model, val_loader, processor)
        current_f1 = metrics["f1"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "metrics": metrics,
        })
        save_json(history, os.path.join(output_dir, "train_history.json"))

        save_json({
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "metrics": metrics,
            "results": val_results
        }, os.path.join(output_dir, f"val_results_epoch_{epoch + 1}.json"))

        # ===================== 保存最新 LoRA 模型（仅权重）=====================
        os.makedirs(latest_adapter_dir, exist_ok=True)
        model.save_pretrained(latest_adapter_dir)   # 只存 LoRA
        processor.save_pretrained(latest_adapter_dir)

        torch.save({
            'epoch': epoch + 1,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'metrics': metrics,
            'best_f1': best_f1,
        }, latest_training_state_path)

        print(f"Latest LoRA adapter saved to {latest_adapter_dir}")
        print(f"Latest training state saved to {latest_training_state_path}")

        # ===================== 保存最优 LoRA 模型 =====================
        if current_f1 > best_f1:
            best_f1 = current_f1

            os.makedirs(best_adapter_dir, exist_ok=True)
            model.save_pretrained(best_adapter_dir)
            processor.save_pretrained(best_adapter_dir)

            torch.save({
                'epoch': epoch + 1,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'metrics': metrics,
                'best_f1': best_f1,
            }, best_training_state_path)

            print(f"Best LoRA adapter saved to {best_adapter_dir}, F1={best_f1:.4f}")
            print(f"Best training state saved to {best_training_state_path}")


def main():
    os.makedirs(args.output_dir, exist_ok=True)

    save_json({
        "args": vars(args),
        "groups": {
            "GroupA": GroupA,
            "GroupB": GroupB,
            "GroupC": GroupC,
        }
    }, os.path.join(args.output_dir, "run_config.json"))

    model_name = args.model_path
    train_relation_prototype = torch.load(args.prototype_path)
    processor = Qwen2VLProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model.register_buffer("plora_embeddings", train_relation_prototype.to(model.device))

    class FixedPeftModel(PeftModel):
        def forward(self, *args, **kwargs):
            kwargs.pop("inputs_embeds", None)
            return super().forward(*args, **kwargs)

    # 保持你原来的 LoRA 逻辑不变
    lora_config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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

    train_dataset = METDataset(data_paths, args.image_dir, processor, 'train')
    val_dataset = METDataset(data_paths, args.image_dir, processor, 'val')
    test_dataset = METDataset(data_paths, args.image_dir, processor, 'test')

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
        train(
            model, train_loader, val_loader, optimizer, scheduler,
            epochs=args.epochs, processor=processor, device=device, output_dir=args.output_dir
        )

    if args.mode in ["eval", "train_eval"]:
        # ===================== 加载 LoRA 权重（按 MRE 风格）=====================
        best_adapter_dir = os.path.join(args.output_dir, "best_adapter")
        latest_adapter_dir = os.path.join(args.output_dir, "latest_adapter")

        if os.path.exists(best_adapter_dir):
            model = PeftModel.from_pretrained(model, best_adapter_dir)
            model = model.to(device)
            print(f"Loaded best LoRA adapter from {best_adapter_dir}")
        elif os.path.exists(latest_adapter_dir):
            model = PeftModel.from_pretrained(model, latest_adapter_dir)
            model = model.to(device)
            print(f"Loaded latest LoRA adapter from {latest_adapter_dir}")

        test_results, category_accuracy = evaluate(model, test_loader, processor)
        metrics = calculate_metrics(test_results)

        save_json({
            "results": test_results,
            "metrics": metrics,
            "category_accuracy": category_accuracy
        }, os.path.join(args.output_dir, "test_results.json"))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
    print("task:", args.task)