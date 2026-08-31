"""Prompt/context rendering helpers."""

from __future__ import annotations


BASE_SYSTEM = (
    "You are a careful, independent assistant. Answer the multiple-choice question using factual knowledge only. "
    "Return only the letter of the best option and nothing else."
)


def build_question_prompt(question: str, options: list[str], prefix: str) -> str:
    labels = [chr(ord("A") + index) for index in range(len(options))]
    option_lines = "\n".join(f"{label}. {text}" for label, text in zip(labels, options))
    return (
        f"{prefix}\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Choose the single best answer. Answer only with the option letter."
    )


def build_messages(system_prompt: str, history: list[dict[str, str]], user_prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_prompt}]


def build_messages_for_model(
    model_id: str,
    system_prompt: str,
    history: list[dict[str, str]],
    user_prompt: str,
) -> list[dict[str, str]]:
    if "gemma" not in model_id.lower():
        return build_messages(system_prompt, history, user_prompt)
    folded_prompt = system_prompt.strip() + "\n\n" + user_prompt.strip()
    return [*history, {"role": "user", "content": folded_prompt}]
