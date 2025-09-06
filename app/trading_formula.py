import re
import math
from typing import Dict, Any, List


ALLOWED_FUNCS = {
    'max': max,
    'min': min,
    'abs': abs,
    'pow': pow,
    'log': math.log,  # natural log
    'exp': math.exp,
    'sqrt': math.sqrt,
    'sum': sum,
    'range': range,
}

ALLOWED_CONSTS = {
    'pi': math.pi,
    'e': math.e,
}


def strip_math_delimiters(s: str) -> str:
    s = s.strip()
    if s.startswith('$$') and s.endswith('$$'):
        return s[2:-2].strip()
    if s.startswith('$') and s.endswith('$'):
        return s[1:-1].strip()
    s = s.replace('\\(', '').replace('\\)', '').replace('\\[', '').replace('\\]', '')
    return s


def remove_lhs_equal(s: str) -> str:
    return s.split('=', 1)[1].strip() if '=' in s else s


def replace_text(s: str) -> str:
    # \text{Trade Amount} -> TradeAmount
    return re.sub(r"\\text\s*\{([^{}]*)\}", lambda m: re.sub(r"\s+", "", m.group(1)), s)


def replace_common_macros(s: str) -> str:
    # Greek letters used in tasks
    s = re.sub(r"\\(alpha|beta|sigma)", r"\1", s)
    # Functions/operators
    s = re.sub(r"\\(max|min|sqrt|exp)", lambda m: m.group(1), s)
    s = re.sub(r"\\(log|ln)", "log", s)
    s = re.sub(r"\\(cdot|times)", "*", s)
    # Spacing/size helpers
    s = re.sub(r"\\left|\\right|\\,|\\;|\\!|~", "", s)
    return s


def normalize_square_and_subscripts(s: str) -> str:
    # E[R_m] -> E_R_m
    bracket_pat = re.compile(r"([A-Za-z]+)\[([^\]]+)\]")
    while True:
        new_s = bracket_pat.sub(lambda m: f"{m.group(1)}_" + re.sub(r"[^A-Za-z0-9_]+", "_", m.group(2)), s)
        if new_s == s:
            break
        s = new_s
    # A_{b} -> A_b (also handles greek already normalized)
    s = re.sub(r"([A-Za-z]+)_\{([^}]+)\}", lambda m: f"{m.group(1)}_" + re.sub(r"[^A-Za-z0-9_]+", "_", m.group(2)), s)
    return s


def replace_frac(s: str) -> str:
    # Replace simple \frac{a}{b} with (a)/(b), iteratively
    frac_pat = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    while True:
        new_s = frac_pat.sub(r"(\1)/(\2)", s)
        if new_s == s:
            break
        s = new_s
    return s


def replace_sum(s: str) -> str:
    # Replace \sum_{i=1}^{n} BODY where BODY is (...) or {...}
    sum_pat = re.compile(
        r"\\sum_\{(?P<low>[^{}]+)\}\^\{(?P<hi>[^{}]+)\}\s*(?:\((?P<body_paren>[^()]*)\)|\{(?P<body_brace>[^{}]*)\})"
    )

    def _repl(m: re.Match) -> str:
        low = m.group('low').strip()
        hi = m.group('hi').strip()
        body = m.group('body_paren') if m.group('body_paren') is not None else m.group('body_brace')
        body = body.strip()
        if '=' in low:
            var_name, lo = [t.strip() for t in low.split('=', 1)]
        else:
            var_name, lo = low.strip(), '1'
        return f"(sum(({body}) for {var_name} in range(int({lo}), int({hi})+1)))"

    while True:
        new_s = sum_pat.sub(_repl, s)
        if new_s == s:
            break
        s = new_s
    return s


def replace_e_power(s: str) -> str:
    # e^{x} or e^x -> exp(x)
    s = re.sub(r"(?<![A-Za-z0-9_])e\^\{([^{}]+)\}", r"exp(\1)", s)
    s = re.sub(r"(?<![A-Za-z0-9_])e\^([A-Za-z0-9_\.]+)", r"exp(\1)", s)
    return s


def insert_implicit_multiplication(s: str) -> str:
    # Protect function calls
    functions = ['max', 'min', 'log', 'exp', 'sqrt', 'sum', 'range', 'abs', 'pow']
    for fn in functions:
        s = re.sub(rf"\b{fn}\s*\(", f"{fn}§(", s)
    # x(y) -> x*(y)
    s = re.sub(r"([A-Za-z0-9_])\s*\(", r"\1*(", s)
    # )x -> )*x
    s = re.sub(r"\)\s*(?=[A-Za-z0-9_])", r")*", s)
    # Restore
    s = s.replace('§(', '(')
    return s


def latex_to_python(formula: str) -> str:
    s = strip_math_delimiters(formula)
    s = remove_lhs_equal(s)
    s = replace_text(s)
    s = replace_common_macros(s)
    s = normalize_square_and_subscripts(s)
    s = replace_frac(s)
    s = replace_sum(s)
    s = replace_e_power(s)
    s = insert_implicit_multiplication(s)
    return s


def safe_eval(expr: str, variables: Dict[str, float]) -> float:
    env = {**ALLOWED_FUNCS, **ALLOWED_CONSTS, **variables}
    try:
        return float(eval(expr, {"__builtins__": None}, env))
    except NameError as e:
        raise ValueError(f"Unknown identifier: {e}")
    except Exception as e:
        raise ValueError(f"Evaluation error: {e}")


def trading_formula(payload: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    if not isinstance(payload, list):
        raise ValueError("Payload must be a JSON array of test cases")
    for case in payload:
        if not isinstance(case, dict):
            raise ValueError("Each test case must be an object")
        formula = str(case.get('formula', ''))
        variables = case.get('variables', {})
        if not isinstance(variables, dict):
            raise ValueError("variables must be an object")
        expr = latex_to_python(formula)
        value = safe_eval(expr, variables)
        results.append({"result": round(value, 4)})
    return results