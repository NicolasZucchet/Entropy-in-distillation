"""Answer checking: symbolic equivalence with `math_verify`, numeric comparison as fallback."""
import re
import threading

NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def _as_number(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _numeric_correct(completion: str, answer: str) -> bool:
    """Compare what the model boxed, else the last number it wrote, with the answer."""
    boxed = BOXED.findall(completion)
    got = _as_number(boxed[-1]) if boxed else None
    if got is None:
        found = NUMBER.findall(completion)
        got = _as_number(found[-1]) if found else None
    want = _as_number(answer)
    return got is not None and want is not None and abs(got - want) < 1e-6


def is_correct(completion: str, answer: str) -> bool:
    """Symbolic equivalence where `math_verify` can parse both sides, numeric otherwise."""
    try:
        from math_verify import parse, verify
    except ImportError:
        return _numeric_correct(completion, answer)
    # `math_verify` bounds sympy with SIGALRM, which only exists on the main thread.
    main = threading.current_thread() is threading.main_thread()
    kw = {} if main else {"parsing_timeout": None}
    vkw = {} if main else {"timeout_seconds": None}
    try:
        gold = parse("\\boxed{" + answer + "}", **kw)
        pred = parse(completion, **kw)
        if gold and pred:
            return bool(verify(gold, pred, **vkw))
    except Exception:
        pass
    return _numeric_correct(completion, answer)


def hits(completions: list[str], answers: list[str]) -> list[bool]:
    return [is_correct(c, a) for c, a in zip(completions, answers)]


def accuracy_stats(correct: list[bool], truncated: list[bool], lengths: list[int]) -> dict:
    """Accuracy, fraction of completions cut by the token budget, and mean length."""
    n = max(len(correct), 1)
    return {"accuracy": sum(correct) / n,
            "truncated_frac": sum(truncated) / n,
            "mean_completion_len": sum(lengths) / n}
