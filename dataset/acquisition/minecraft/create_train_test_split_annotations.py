import random

import pandas as pd

if __name__ == "__main__":

    output_path = "dataset/acquisition/minecraft/minecraft_annotations/splits.csv"
    sequences_count = 248

    validation_portion = 0.15
    test_portion = 0.15

    validation_sequences = int(sequences_count * validation_portion)
    test_sequences = int(sequences_count * test_portion)
    train_sequences = sequences_count - validation_sequences - test_sequences

    # Creates a shuffled list of sequences
    sequences = list(range(sequences_count))
    random.shuffle(sequences)
    sequences_annotations = ["train"] * train_sequences + ["validation"] * validation_sequences + ["test"] * test_sequences

    dataframe = pd.DataFrame(list(zip(sequences, sequences_annotations)), columns =['sequence', 'split'])
    dataframe.to_csv(output_path, index=False)



