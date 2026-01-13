from pathlib import Path

import fasttext

repo_root = Path(__file__).resolve().parents[2]
model_path = repo_root / "experiments" / ".cache" / "lid.176.bin"


def main():
    print(f"Loading model from {model_path}...")
    model = fasttext.load_model(str(model_path))

    samples = [
        "This is a clear English sentence about artificial intelligence.",
        "Esto es una frase en español.",
        "AI and Machine Learning are cool.",
        "Check out this link: https://t.co/xyz",
        "ArtificialIntelligence is the future #AI",
    ]

    print("\nTesting predictions:")
    for text in samples:
        clean_text = text.replace("\n", " ")
        labels, scores = model.predict(clean_text)
        print(f"Text: '{text}'")
        print(f"  Label: {labels[0]}, Score: {scores[0]}")
        print(f"  Is English? {labels[0] == '__label__en'}")
        print("-" * 40)


if __name__ == "__main__":
    main()
