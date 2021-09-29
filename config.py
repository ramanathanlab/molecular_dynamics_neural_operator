"""Generate a configuration file to input to the graph_kernel python file"""
from pathlib import Path

# from typing import Optional
from mdh.models.config import BaseSettings


class GANModelConfig(BaseSettings):

    # File paths
    # Path to training file
    dataset_fasta_input_path: Path = Path(
        "/lambda_stor/projects/MDH-design/mdh_095_threshold.fasta"
    )
    # Blast against a database or a subject file
    blast_against_type: str = "subject"
    # Path to directory where trainer should write to (cannot already exist)
    output_path: Path = Path("results/")

    # Length of sequences (will apply padding to shorter seqs)
    input_length: int = 1024
    # Fraction of training data to use
    split_pct: float = 0.8
    # Training batch size
    batch_size: int = 32
    # Generator learning rate:
    generator_learning_rate: float = 0.00001
    # Discriminator learning rate
    discriminator_learning_rate: float = 0.00001
    # Dimensions for input noise
    noise_dims: int = 128
    # Total amount of steps:
    max_epochs: int = 150
    # number of sequences tot predict
    num_seqs_to_predict: int = 1000

    # Random seed for training
    seed: int = 333
    # WANBD Project Name
    wandb_project: str = "mdh_gan_training"
    # Specify if using codon or nucleotide modality
    codon_or_nucleotide: str = "codon"
    # Specify where to place the
    cached_dataset_location: Path = Path("/tmp/mzvyagin/mdh_dataset.pkl")
    blast_validation_location: Path = Path("/tmp/mzvyagin/validation_blast_set.fasta")


if __name__ == "__main__":
    GANModelConfig().dump_yaml("gan_template.yaml")