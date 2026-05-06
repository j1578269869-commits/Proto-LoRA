import hashlib
import random
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import json
import os
from peft import LoraConfig, get_peft_model, PeftModel
from argparse import ArgumentParser
from config.loader import apply_config, load_config

# ============================================================
# 1. 基础设置
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument("--config", type=str, default="", help="Path to config YAML/JSON file")
parser.add_argument("--seed", type=int, default=97)  # 16,67,97
parser.add_argument("--task", type=int, choices=[1, 2, 3], default=1)

parser.add_argument("--mode", type=str, default="train_eval", choices=["train", "eval", "train_eval"])
parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--prototype_path", type=str, default="")
parser.add_argument("--train_path", type=str, default="")
parser.add_argument("--val_path", type=str, default="")
parser.add_argument("--test_path", type=str, default="")
parser.add_argument("--image_dir", type=str, default="")
parser.add_argument("--output_dir", type=str, default="./checkpoints_blip2_met")

parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)

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


# ============================================================
# 2. 工具函数
# ============================================================
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class FixedPeftModel(PeftModel):
    def forward(self, *args, **kwargs):
        kwargs.pop("inputs_embeds", None)
        return super().forward(*args, **kwargs)


def get_lora_config():
    return LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=r"(qformer|t5_model)\..*\.query|.*\.value",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
    )


def build_base_model_and_processor(model_path, prototype_path):
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    processor = Blip2Processor.from_pretrained(model_path)

    train_relation_prototype = torch.load(prototype_path, map_location="cpu")
    model.register_buffer("plora_embeddings", train_relation_prototype)

    # 保持你原来的冻结策略不变
    for name, param in model.named_parameters():
        if "vision_model" in name or "qformer" in name:
            param.requires_grad = False

    return model, processor


def wrap_with_lora(base_model):
    lora_config = get_lora_config()
    model = get_peft_model(base_model, lora_config)
    model = FixedPeftModel(model.model, peft_config=lora_config)
    model = model.to(device)
    return model


def save_lora_only(model, save_dir, processor=None):
    """
    这里改成直接用 PeftModel.save_pretrained() 保存 adapter，
    不再手动抽取 state_dict，避免保存/加载链路不一致。
    """
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    if processor is not None:
        processor.save_pretrained(save_dir)
    print(f"✅ LoRA adapter saved to {save_dir}")


def load_lora_for_eval(adapter_dir, model_path, prototype_path):
    """
    关键修复：
    评测时不要把 adapter 再加载到一个已经 get_peft_model 包过的模型上，
    而是重新构建一个“干净的 base model”，再加载 adapter。
    """
    base_model, processor = build_base_model_and_processor(model_path, prototype_path)
    model = FixedPeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    model = model.to(device)
    return model, processor


# ============================================================
# 3. 数据集
# ============================================================
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

        if self.mode == "train" and args.train_max_samples > 0:
            self.dataset = self.dataset[:args.train_max_samples]
        elif self.mode == "val" and args.val_max_samples > 0:
            self.dataset = self.dataset[:args.val_max_samples]
        elif self.mode == "test" and args.test_max_samples > 0:
            self.dataset = self.dataset[:args.test_max_samples]

    def _load_data(self, data_paths, allowed_types):
        """
        这里先保持你原来版本的逻辑不动，只修“新版结果变 0”的主因。
        """
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
                            "entity_types": sorted(valid_entity_types)
                        })

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
        dynamic_options_str = "Options:\n" + "\n".join(
            [f"{rel}" for i, rel in enumerate(current_options)]
        )

        input_text = (
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

    image_inputs = processor(images=images, return_tensors="pt")
    text_inputs = processor(
        text=input_texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    with processor.tokenizer.as_target_tokenizer():
        labels_encoding = processor.tokenizer(
            text=target_texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=15,
        )
        labels = labels_encoding.input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "pixel_values": image_inputs.pixel_values,
        "input_ids": text_inputs.input_ids,
        "attention_mask": text_inputs.attention_mask,
        "labels": labels,
        "target_text": target_texts
    }


# ============================================================
# 4. 指标与评估
# ============================================================
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

    f1 = 0.0
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
    model.eval()
    total_loss = 0
    results = []

    if task == 1:
        valid_categories = set(["Location", "Event", "Book", "Music"])
    elif task == 2:
        valid_categories = set(["Organization", "Country", "APP", "Movie"])
    elif task == 3:
        valid_categories = set(["People", "Site", "Building", "Currency"])

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", total=len(val_loader))
        for batch in progress_bar:
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device)
            }

            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            generated_outputs = model.generate(
                pixel_values=batch["pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )

            pred_texts = processor.batch_decode(generated_outputs, skip_special_tokens=True)
            input_texts = processor.batch_decode(batch["input_ids"], skip_special_tokens=True)

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

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    metrics = calculate_metrics(results)
    return avg_loss, metrics, results


def evaluate(model, test_loader, processor):
    model.eval()
    results = []
    category_correct = {}
    category_total = {}
    confusion_matrix = {}

    if task == 1:
        valid_categories = set(["Organization", "Country", "APP", "Movie"])
    elif task == 2:
        valid_categories = set(["People", "Site", "Building", "Currency"])
    elif task == 3:
        valid_categories = set(["Location", "Event", "Book", "Music"])

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
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


# ============================================================
# 5. 训练
# ============================================================
def train(model, train_loader, val_loader, optimizer, scheduler, epochs, processor, output_dir="checkpoints"):
    torch.cuda.empty_cache()
    os.makedirs(output_dir, exist_ok=True)

    latest_adapter_dir = os.path.join(output_dir, "latest_adapter")
    best_adapter_dir = os.path.join(output_dir, "best_adapter")

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
                "labels": batch["labels"].to(device)
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

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
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

        save_lora_only(model, latest_adapter_dir, processor)

        torch.save({
            'epoch': epoch + 1,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'metrics': metrics,
            'best_f1': best_f1,
        }, latest_training_state_path)

        print(f"Latest training state saved to {latest_training_state_path}")

        if current_f1 > best_f1:
            best_f1 = current_f1
            save_lora_only(model, best_adapter_dir, processor)

            torch.save({
                'epoch': epoch + 1,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'metrics': metrics,
                'best_f1': best_f1,
            }, best_training_state_path)

            print(f"Best adapter saved to {best_adapter_dir}, F1={best_f1:.4f}")
            print(f"Best training state saved to {best_training_state_path}")


# ============================================================
# 6. 主函数
# ============================================================
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

    # 训练用模型
    base_model, processor = build_base_model_and_processor(args.model_path, args.prototype_path)
    model = wrap_with_lora(base_model)
    model.print_trainable_parameters()

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
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            processor=processor,
            output_dir=args.output_dir
        )

    # =========================
    # 关键修复点 1：
    # train_eval 模式下，训练完成后直接评测当前内存里的模型
    # =========================
    if args.mode == "train_eval":
        print("✅ train_eval 模式：直接使用当前训练后的模型进行测试，不重新加载 adapter。")
        test_results, category_accuracy = evaluate(model, test_loader, processor)
        metrics = calculate_metrics(test_results)

        save_json({
            "results": test_results,
            "metrics": metrics,
            "category_accuracy": category_accuracy
        }, os.path.join(args.output_dir, "test_results.json"))

    # =========================
    # 关键修复点 2：
    # 纯 eval 模式下，重新构建干净 base model，再加载 adapter
    # =========================
    if args.mode == "eval":
        best_adapter_dir = os.path.join(args.output_dir, "best_adapter")
        latest_adapter_dir = os.path.join(args.output_dir, "latest_adapter")

        adapter_dir = None
        if os.path.exists(best_adapter_dir):
            adapter_dir = best_adapter_dir
            print(f"✅ Found best adapter: {best_adapter_dir}")
        elif os.path.exists(latest_adapter_dir):
            adapter_dir = latest_adapter_dir
            print(f"✅ Found latest adapter: {latest_adapter_dir}")
        else:
            raise FileNotFoundError("未找到可用于 eval 的 adapter，请先训练或检查输出目录。")

        eval_model, eval_processor = load_lora_for_eval(
            adapter_dir=adapter_dir,
            model_path=args.model_path,
            prototype_path=args.prototype_path
        )

        test_results, category_accuracy = evaluate(eval_model, test_loader, eval_processor)
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