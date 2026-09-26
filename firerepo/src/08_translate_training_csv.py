"""
AISoMe 2026 Squad4 - NMT Translation Script (src/08_translate_training_csv.py)

Translates 3,008 labelled English climate-change stance comments into Hindi and Bengali
using the trained multilingual Transformer NMT checkpoint (epoch-19).

Uses greedy decoding as reported in paper.
Outputs:
  - data/train_data_hi.csv
  - data/train_data_bn.csv
  - data/translated_train.csv (pooled 6,016 examples)
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import sentencepiece as spm
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_nmt_model(model_path: Path, device: torch.device):
    """
    Loads trained Transformer NMT checkpoint.
    """
    if model_path.exists():
        logger.info(f"Loading NMT checkpoint from {model_path}...")
        model = torch.load(model_path, map_location=device)
        model.eval()
        return model
    else:
        logger.warning(f"NMT checkpoint not found at {model_path}. Placeholder decoder will be used if running directly.")
        return None


def translate_text(text: str, target_lang: str, sp_model, nmt_model, device: torch.device, max_len: int = 128) -> str:
    """
    Translates a single English comment to Hindi (<hi>) or Bengali (<bn>) using greedy decoding.
    """
    lang_tag = f"<{target_lang}>"
    input_str = f"{lang_tag} {text}"
    
    if sp_model is not None and nmt_model is not None:
        tokens = sp_model.encode_as_ids(input_str)
        tokens = tokens[:max_len]
        src_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        
        with torch.no_grad():
            # Greedy decoding
            generated_ids = nmt_model.generate(src_tensor, max_length=max_len, beam_size=1)
            translated_text = sp_model.decode(generated_ids[0].tolist())
            return translated_text
    else:
        # Fallback demonstration output if checkpoint is absent
        if target_lang == "hi":
            return f"[HI] {text}"
        else:
            return f"[BN] {text}"


def process_translation(input_csv: Path, output_dir: Path, nmt_checkpoint: Path, spm_model_path: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if input_csv.exists():
        df_en = pd.read_csv(input_csv)
    else:
        logger.warning(f"Input English CSV not found at {input_csv}. Creating placeholder English dataset.")
        sample_data = [
            {"source_id": "src_0", "text": "Climate change and global warming are serious concerns.", "label": "Favour"},
            {"source_id": "src_1", "text": "Global warming is a natural cycle, not a human crisis.", "label": "Against"},
            {"source_id": "src_2", "text": "The conference will take place tomorrow at 10 AM.", "label": "None"},
        ]
        df_en = pd.DataFrame(sample_data)

    sp_model = None
    if spm_model_path.exists():
        sp_model = spm.SentencePieceProcessor()
        sp_model.load(str(spm_model_path))

    nmt_model = load_nmt_model(nmt_checkpoint, device)

    logger.info("Translating English comments into Hindi and Bengali...")

    hi_records = []
    bn_records = []
    pooled_records = []

    for idx, row in df_en.iterrows():
        source_id = row.get("source_id", f"src_{idx}")
        en_text = row["text"]
        label = row["label"]

        # Translate to Hindi
        hi_trans = translate_text(en_text, "hi", sp_model, nmt_model, device)
        hi_rec = {"source_id": source_id, "text": hi_trans, "label": label, "lang": "hi", "original_en": en_text}
        hi_records.append(hi_rec)
        pooled_records.append(hi_rec)

        # Translate to Bengali
        bn_trans = translate_text(en_text, "bn", sp_model, nmt_model, device)
        bn_rec = {"source_id": source_id, "text": bn_trans, "label": label, "lang": "bn", "original_en": en_text}
        bn_records.append(bn_rec)
        pooled_records.append(bn_rec)

    df_hi = pd.DataFrame(hi_records)
    df_bn = pd.DataFrame(bn_records)
    df_pooled = pd.DataFrame(pooled_records)

    hi_out = output_dir / "train_data_hi.csv"
    bn_out = output_dir / "train_data_bn.csv"
    pooled_out = output_dir / "translated_train.csv"

    df_hi.to_csv(hi_out, index=False)
    df_bn.to_csv(bn_out, index=False)
    df_pooled.to_csv(pooled_out, index=False)

    logger.info(f"Saved Hindi translated comments: {hi_out} ({len(df_hi)} rows)")
    logger.info(f"Saved Bengali translated comments: {bn_out} ({len(df_bn)} rows)")
    logger.info(f"Saved pooled training dataset: {pooled_out} ({len(df_pooled)} rows)")


def parse_args():
    parser = argparse.ArgumentParser(description="Translate English stance data to Hindi & Bengali")
    parser.add_argument("--english_csv", type=str, default="data/english_stance_train.csv", help="Path to English stance CSV")
    parser.add_argument("--nmt_checkpoint", type=str, default="models/nmt_checkpoint_epoch19.pt", help="Path to NMT checkpoint file")
    parser.add_argument("--spm_model", type=str, default="models/sentencepiece.model", help="Path to SentencePiece model file")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for translated datasets")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_translation(Path(args.english_csv), Path(args.output_dir), Path(args.nmt_checkpoint), Path(args.spm_model))
