"""Check that BAAI/bge-m3 works on this machine — Stage 2's semantic half.

Run it from the project root:

    python check_embeddings.py

No API key is involved: Stage 2 runs the model locally, unlike Stages 1 and 3.
The first run downloads ~2 GB from HuggingFace into ~/.cache/huggingface and takes
a few minutes; every run after that loads from the cache in seconds.

It calls the project's own `semantic_scores`, so a pass means the code path Stage 2
actually uses works — not a parallel implementation that happens to work.
"""

import time

from screening.stage2_ranking.embeddings import get_embedding_model, semantic_scores

JD = "Python backend development: building server-side services"

# One obvious match, one that never says "Python", one in French, one from another
# planet. Their relative order is the whole point of the semantic half.
CANDIDATES = [
    ("literal match", "Skills: Python, Django. Backend Engineer: built server-side APIs in Python."),
    ("paraphrase", "Skills: server-side programming. Software Engineer: web services and REST APIs."),
    ("French, same role", "Compétences : développement backend. Ingénieur logiciel : services web côté serveur."),
    ("unrelated", "Skills: Pastry, cake decorating. Pastry Chef: prepared desserts and viennoiseries."),
]


def main() -> int:
    print("Loading BAAI/bge-m3 (the first run downloads ~2 GB)...")
    started = time.perf_counter()
    model = get_embedding_model()
    print(f"  loaded in {time.perf_counter() - started:.1f}s")
    print(f"  embedding dimension : {model.get_sentence_embedding_dimension()}")
    print(f"  max sequence length : {model.max_seq_length} tokens")

    print(f"\nJD: {JD}\n")
    started = time.perf_counter()
    scores = semantic_scores(JD, [text for _label, text in CANDIDATES])
    elapsed = time.perf_counter() - started
    print(f"Encoded {len(CANDIDATES)} candidates in {elapsed:.1f}s\n")

    for (label, _text), score in zip(CANDIDATES, scores):
        bar = "#" * max(0, round(score * 40))
        print(f"  {score:+.3f}  {bar:<40}  {label}")

    literal, paraphrase, french, unrelated = scores
    checks = [
        ("scores are cosine similarities in [-1, 1]", all(-1.0 <= s <= 1.0 for s in scores)),
        ("the literal match scores highest", literal == max(scores)),
        ("the paraphrase beats the unrelated CV", paraphrase > unrelated),
        ("French tracks English for the same role", french > unrelated and french > 0.5),
    ]

    print()
    for description, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {description}")

    failed = [description for description, ok in checks if not ok]
    print()
    if failed:
        print(f"{len(failed)} check(s) failed: the model loaded but is not ranking the way")
        print("Stage 2 depends on. Ranking results would be unreliable.")
        return 1
    print("All checks passed — Stage 2's semantic ranking works on this machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
