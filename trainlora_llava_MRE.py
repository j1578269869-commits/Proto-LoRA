import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import LlavaForConditionalGeneration, LlavaProcessor, get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import os
from peft import LoraConfig, get_peft_model, PeftModel
from argparse import ArgumentParser
from config.loader import apply_config, load_config

# version把数据集合并，实验设置完全统一
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument("--config", type=str, default="", help="Path to config YAML/JSON file")
parser.add_argument("--seed", type=int, default=25)
parser.add_argument("--mode", type=str, default="train_eval", choices=["train", "eval", "train_eval"])
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--prototype_path", type=str, default="")
parser.add_argument("--train_path", type=str, default="")
parser.add_argument("--val_path", type=str, default="")
parser.add_argument("--test_path", type=str, default="")
parser.add_argument("--image_base_path", type=str, default="")
parser.add_argument("--save_dir", type=str, default="./checkpoints_llava_mre")
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

if not args.model_name or not args.prototype_path:
    raise ValueError("Please specify --model_name and --prototype_path either directly or via --config")

if not args.train_path or not args.val_path or not args.test_path:
    raise ValueError("Please specify --train_path, --val_path, and --test_path either directly or via --config")

if not args.image_base_path:
    raise ValueError("Please specify --image_base_path either directly or via --config")


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

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

rel_lower = {rel.lower(): rel for rel in candidate_relations}


def split_types(types):
    filtered_types = ["Other", "None", "none", "NA", "P0"]
    types = [ele for ele in types if ele not in filtered_types]
    types_sorted = sorted(types)
    np.random.seed(args.seed)
    types_shuffled = types_sorted.copy()
    np.random.shuffle(types_shuffled)
    n_types = len(types_shuffled)
    avg_n_types = n_types // 3
    train_types = types_shuffled[: avg_n_types + 1]
    val_types = types_shuffled[avg_n_types + 1: avg_n_types * 2 + 1]
    test_types = types_shuffled[avg_n_types * 2 + 1:]
    return train_types, val_types, test_types


class MNREDataset(Dataset):
    _train_types = None
    _val_types = None
    _test_types = None
    _has_split = False

    def __init__(self, data_paths, image_base_path, processor, mode='train'):
        self.dataset = []
        self.image_base_path = image_base_path
        self.processor = processor
        self.mode = mode

        if not MNREDataset._has_split:
            all_data = []
            for data_path in data_paths:
                with open(data_path, 'r', encoding='utf-8') as f:
                    all_data.extend(json.load(f))

            all_types = list(set(item['relation'] for item in all_data))
            train_types, val_types, test_types = split_types(all_types)

            MNREDataset._train_types = set(train_types)
            MNREDataset._val_types = set(val_types)
            MNREDataset._test_types = set(test_types)
            MNREDataset._has_split = True

            print(f"✅ Split done with seed={args.seed}:")
            print(f"   Train types ({len(train_types)}): {sorted(train_types)}")
            print(f"   Val types   ({len(val_types)}): {sorted(val_types)}")
            print(f"   Test types  ({len(test_types)}): {sorted(test_types)}")

        for data_path in data_paths:
            with open(data_path, 'r', encoding='utf-8') as f:
                temp_dataset = json.load(f)
                self.dataset.extend(temp_dataset)

        if self.mode == 'train':
            self.dataset = [item for item in self.dataset if item['relation'] in MNREDataset._train_types]
            self.available_relations = sorted(relation_map[t] for t in MNREDataset._train_types if t in relation_map)
        elif self.mode == 'val':
            self.dataset = [item for item in self.dataset if item['relation'] in MNREDataset._val_types]
            self.available_relations = sorted(relation_map[t] for t in MNREDataset._val_types if t in relation_map)
        elif self.mode == 'test':
            self.dataset = [item for item in self.dataset if item['relation'] in MNREDataset._test_types]
            self.available_relations = sorted(relation_map[t] for t in MNREDataset._test_types if t in relation_map)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if self.mode == 'train' and args.train_max_samples > 0:
            self.dataset = self.dataset[:args.train_max_samples]
        elif self.mode == 'val' and args.val_max_samples > 0:
            self.dataset = self.dataset[:args.val_max_samples]
        elif self.mode == 'test' and args.test_max_samples > 0:
            self.dataset = self.dataset[:args.test_max_samples]

        self.options_str = "Options:\n" + "\n".join([
            f"{i + 1}. {rel}" for i, rel in enumerate(self.available_relations)
        ])

        print(f"📊 {self.mode} dataset loaded: {len(self.dataset)} samples, "
              f"{len(self.available_relations)} candidate relations")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = os.path.join(self.image_base_path, item["img_id"])
        image = Image.open(image_path).convert("RGB")

        input_text = (
            f"Image: <image>\n"
            "Identify the relationship between entities:\n"
            f"Text: {' '.join(item['token'])}\n"
            f"Head Entity: {item['h']['name']} (Position: {item['h']['pos']})\n"
            f"Tail Entity: {item['t']['name']} (Position: {item['t']['pos']})\n"
            + self.options_str + "\n"
            "Answer only with the relation name, no other text.\n"
            "Relation:"
        )

        target_text = relation_map.get(item["relation"])

        return {
            "image": image,
            "input_text": input_text,
            "target_text": target_text,
            "id": item.get("id", f"item_{idx}")
        }


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


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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

    f1 = 0
    avg_prec = avg_prec / len(set(predicted_labels[:i]))
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

                true_rel = batch["target_texts"][i]

                results.append({
                    "id": batch["ids"][i],
                    "prediction": prediction,
                    "true_relation": true_rel,
                    "correct": prediction == true_rel
                })

            del inputs, outputs, loss
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(val_loader)
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
                outputs = model.generate(**inputs)

            pred_texts = processor.batch_decode(outputs, skip_special_tokens=True)
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

                true_rel = batch["target_texts"][i]

                if true_rel not in category_total:
                    category_total[true_rel] = 0
                    category_correct[true_rel] = 0
                    confusion_matrix[true_rel] = {}
                category_total[true_rel] += 1

                if prediction == true_rel:
                    category_correct[true_rel] += 1
                else:
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
        category_accuracy[category] = correct / total if total > 0 else 0

    return results, category_accuracy


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


def train_with_periodic_evaluation(model, train_loader, val_loader, test_loader, optimizer, scheduler, epochs, processor, device, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    # 保存路径：只存 LoRA
    best_lora_dir = os.path.join(save_dir, "best_lora")
    latest_lora_dir = os.path.join(save_dir, "latest_lora")

    accumulation_steps = 4
    scaler = torch.cuda.amp.GradScaler()
    all_test_metrics = []

    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*50}")

        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training")

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

            if (i + 1) % accumulation_steps == 0:
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

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

        val_loss, val_metrics = validate(model, val_loader, processor)
        print(f"Validation Loss: {val_loss:.4f}, Metrics: {val_metrics}")

        print(f"\n🔄 正在执行第 {epoch + 1} 轮测试...")
        test_results, category_accuracy = evaluate(model, test_loader, processor)
        test_metrics = calculate_metrics(test_results)

        save_test_result(test_results, test_metrics, epoch=epoch+1, save_dir=save_dir)

        all_test_metrics.append({
            "epoch": epoch + 1,
            "metrics": test_metrics,
            "val_loss": val_loss,
            "train_loss": avg_train_loss
        })

        # ===================== 保存最新 LoRA =====================
        model.save_pretrained(latest_lora_dir)
        processor.save_pretrained(latest_lora_dir)

        # ===================== 保存最优 LoRA =====================
        current_f1 = test_metrics["f1"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch + 1

            # 只保存 LoRA
            model.save_pretrained(best_lora_dir)
            processor.save_pretrained(best_lora_dir)

            print(f"🎉 最佳 LoRA 已保存：F1={best_f1:.4f}")
            print(f"💾 路径：{best_lora_dir}")

    print(f"\n{'='*60}")
    print("📊 所有轮次测试结果汇总")
    print(f"{'='*60}")
    for record in all_test_metrics:
        epoch = record["epoch"]
        f1 = record["metrics"]["f1"]
        acc = record["metrics"]["accuracy"]
        prec = record["metrics"]["precision"]
        rec = record["metrics"]["recall"]
        print(f"Epoch {epoch:2d}: F1={f1:.4f}, Acc={acc:.2%}, Prec={prec:.4f}, Rec={rec:.4f}")

    print(f"\n🏆 最佳结果：Epoch {best_epoch} | F1={best_f1:.4f}")
    return all_test_metrics, best_epoch, best_f1


def main():
    os.makedirs(args.save_dir, exist_ok=True)

    model_name = args.model_name
    train_relation_prototype = torch.load(args.prototype_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    processor = LlavaProcessor.from_pretrained(model_name)

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

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
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
    train_dataset = MNREDataset(data_paths, args.image_base_path, processor, 'train')
    val_dataset = MNREDataset(data_paths, args.image_base_path, processor, 'val')
    test_dataset = MNREDataset(data_paths, args.image_base_path, processor, 'test')

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

    save_json(vars(args), os.path.join(args.save_dir, "run_config.json"))

    if args.mode in ["train", "train_eval"]:
        all_metrics, best_epoch, best_f1 = train_with_periodic_evaluation(
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

    # ===================== 测试阶段：加载 LoRA =====================
    if args.mode in ["eval", "train_eval"]:
        best_lora_path = os.path.join(args.save_dir, "best_lora")
        if os.path.exists(best_lora_path):
            model = PeftModel.from_pretrained(model, best_lora_path)
            print("✅ 已加载最佳 LoRA 模型")

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