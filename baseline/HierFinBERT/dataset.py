"""
dataset.py - PyTorch Dataset for Hierarchical BERT.
"""
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
import pandas as pd
from transformers import PreTrainedTokenizer

from segmentation import greedy_sentence_chunking

class HierarchicalDataset(Dataset):
    """
    PyTorch Dataset for handling hierarchically segmented text data.

    This class takes a DataFrame of transcripts, segments each transcript using
    the greedy_sentence_chunking function, and tokenizes the segments for use
    in a hierarchical model like Hierarchical BERT.
    """

    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer, max_segment_len: int, threshold: float = 0.05):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'transcript' and 'future_3bday_cum_return' columns.
            tokenizer (PreTrainedTokenizer): The tokenizer for the model.
            max_segment_len (int): The 'L' hyperparameter for chunking (e.g., 32, 64, 128).
            threshold (float): The threshold for classifying returns into Up or Down.
        """
        self.tokenizer = tokenizer
        self.max_segment_len = max_segment_len
        
        # --- Filter out neutral cases and create binary labels ---
        # 0: Down, 1: Up
        filtered_df = df[(df["future_3bday_cum_return"] > threshold) | (df["future_3bday_cum_return"] < -threshold)].copy()
        
        filtered_df['label'] = filtered_df['future_3bday_cum_return'].apply(lambda r: 1 if r > threshold else 0)
        self.labels = torch.tensor(filtered_df['label'].values, dtype=torch.long)
        
        # --- Pre-process all transcripts to find max segments and store tokenized chunks ---
        all_tokenized_chunks = []
        self.segment_lengths = []
        max_segments = 0

        # First pass: segment and tokenize to find the max number of segments
        for transcript in filtered_df["transcript"]:
            transcript_str = str(transcript) if pd.notna(transcript) else ""
            chunks = greedy_sentence_chunking(transcript_str, self.tokenizer, self.max_segment_len)
            self.segment_lengths.append(len(chunks))
            if len(chunks) > max_segments:
                max_segments = len(chunks)
            all_tokenized_chunks.append(chunks)

        # Second pass: pad all transcripts to the max number of segments
        self.processed_transcripts = []
        for chunks in all_tokenized_chunks:
            tokenized_output = self.tokenizer(
                chunks,
                max_length=self.max_segment_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Manually pad the number of segments
            num_to_pad = max_segments - len(chunks)
            if num_to_pad > 0:
                pad_shape_ids = (num_to_pad, self.max_segment_len)
                pad_shape_mask = (num_to_pad, self.max_segment_len)
                
                padded_ids = torch.cat([
                    tokenized_output["input_ids"],
                    torch.zeros(pad_shape_ids, dtype=torch.long)
                ], dim=0)
                
                padded_mask = torch.cat([
                    tokenized_output["attention_mask"],
                    torch.zeros(pad_shape_mask, dtype=torch.long)
                ], dim=0)
                
                tokenized_output = {"input_ids": padded_ids, "attention_mask": padded_mask}

            self.processed_transcripts.append(tokenized_output)

    def __len__(self) -> int:
        """Returns the total number of samples (transcripts)."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns one sample of data, which includes the tokenized chunks
        of a transcript and its corresponding label.
        """
        return {
            "input_ids": self.processed_transcripts[idx]["input_ids"],
            "attention_mask": self.processed_transcripts[idx]["attention_mask"],
            "labels": self.labels[idx],
            "segment_lengths": self.segment_lengths[idx]
        } 