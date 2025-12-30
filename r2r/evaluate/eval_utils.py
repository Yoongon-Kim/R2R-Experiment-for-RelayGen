from typing import Dict, List, Tuple, Callable, Any, Optional, Union
import re
from datasets import load_dataset
import json
import base64
import zlib
import pickle
import numpy as np
from math import isclose
import regex
from latex2sympy2 import latex2sympy
## USED FOR GPQA ###

# ===== Helper functions from RouteLRM for robust answer extraction and comparison =====

def _fix_fracs(string):
    """Fix LaTeX fractions to proper format."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    """Convert a/b format to LaTeX fraction."""
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    """Fix sqrt formatting."""
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def strip_string(string, skip_unit=False):
    """Normalize and clean LaTeX strings for comparison."""
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{."
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string = string.replace("'", "")
    string = string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def parse_digits(num):
    """Parse string to number, handling percentages."""
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num):
    """Check if string can be parsed as digit."""
    return parse_digits(num) is not None


def numeric_equal(prediction: float, reference: float):
    """Check if two numbers are equal within tolerance."""
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    """Check if two expressions are symbolically equal."""
    from sympy import simplify, N
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr

    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal

    This is the main comparison function from RouteLRM.
    """
    if prediction is None or reference is None:
        return False

    # String comparison
    if str(prediction).strip().lower() == str(reference).strip().lower():
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                    for i in range(len(pred_parts))
                ]
            ):
                return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close
        ):
            return True

    # symbolic equal with sympy
    if symbolic_equal(prediction, reference):
        return True

    return False


def find_boxed_content(text: str) -> str:
    """Extract content from \\boxed{...} with proper brace matching."""
    ans = text.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
        return a
    else:
        a = ans.split("$")[0].strip()
        return a

# ===== End of RouteLRM helper functions =====

QUERY_TEMPLATE_MULTICHOICE = """
What is the correct answer to the following problem? Please reason step by step. 
Separate logical reasoning steps with two newline characters (\n\n).
Put the final answer **strictly** in the format \\boxed{{X}}, where X is a single letter (A, B, C, or D).

**Example output:** \\boxed{{A}}

Problem: {Question}
Choices:
(A) {A}
(B) {B}
(C) {C}
(D) {D}
""".strip()

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"


## USED FOR MMLU-PRO ###
choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

def check_answer_correctness(predicted: str, actual: str, answer_type: str) -> bool:
    """Check if the predicted answer is correct based on the answer type.

    Uses RouteLRM's robust math_equal function for boxed answers to handle
    LaTeX expressions like \\frac{200}{1}, symbolic math, and numerical comparisons.
    """
    if answer_type == "boxed":
        # Use RouteLRM's robust math comparison that handles:
        # - LaTeX expressions (e.g., \frac{200}{1})
        # - Symbolic equality (e.g., x+1 vs 1+x)
        # - Numerical equality with tolerance
        # First normalize both strings
        predicted_normalized = strip_string(predicted)
        actual_normalized = strip_string(actual)
        return math_equal(predicted_normalized, actual_normalized)
    elif answer_type == "multiple_choice":
        # For multiple choice, compare uppercase letters
        return predicted.upper() == actual.upper()
    elif answer_type == "code":
        # For code answers, we'll need to run test cases - for now just check if code is not empty
        # TODO: Implement actual code testing logic
        return bool(predicted.strip())
    elif answer_type == "mmlu-multiple-choice":
        return predicted.upper() == actual.upper()
    elif answer_type == "livecodebench":
        return True
    else:
        raise ValueError(f"Unsupported answer type for correctness check: {answer_type}")

def get_answer_extractor(dataset_type: str) -> Callable:
    """Return the appropriate answer extraction function based on dataset type."""
    extractors = {
        "boxed": extract_boxed_answer,
        "multiple_choice": extract_multiple_choice_answer,
        "livecodebench": dummy_extract_code_answer,
        "mmlu-multiple-choice": extract_mmlu_pro_answer
    }
    
    if dataset_type in extractors:
        return extractors[dataset_type]
    else:
        raise ValueError(f"Unsupported dataset answer type: {dataset_type}")
    
def extract_boxed_answer(text: str) -> Tuple[str, bool]:
    """Extract answer from \\boxed{...} with proper brace matching and normalization.

    Uses RouteLRM's approach:
    1. Properly handles nested braces in LaTeX (e.g., \\boxed{\\frac{200}{1}})
    2. Normalizes the extracted answer using strip_string
    3. Returns normalized answer that can be compared with math_equal
    """
    if "boxed" not in text:
        return "", False

    # Use RouteLRM's find_boxed_content which properly handles nested braces
    answer = find_boxed_content(text)

    if answer:
        # Normalize the answer for comparison
        normalized_answer = strip_string(answer)
        return normalized_answer, True

    return "", False

def extract_multiple_choice_answer(text: str) -> Tuple[str, bool]:
    # """Extract multiple-choice answer (A, B, C, or D) from the generated text."""
    # # Look for answer statements like "The answer is A" or "I choose B"
    
    # match = re.search(ANSWER_PATTERN_MULTICHOICE, text)
    # if match:
    #     choice = match.group(1).upper()  # Convert to uppercase
    #     return choice, True
    
    # return "", False
    """Extract answer from \boxed{...} and check if it's a valid number.""" # Yoon added this to correctly get answer for gpqa dataset becuase Yoon changed the prompt for gpqa
    pattern = r"\\boxed{([^}]*)}"
    match = re.search(pattern, text)
    if match:
        answer = match.group(1).strip()
        # Try to convert to int, return None if fails
        try:
            return answer, True
        except:
            return answer, False
    return "", False

def dummy_extract_code_answer(text: str) -> Tuple[str, bool]:
    """Dummy function to extract code from the generated text."""
    return text, False

def extract_code_answer(text: str) -> Tuple[str, bool]:
    """Extract Python code from the generated text."""
    # Look for code blocks marked with ```python or ``` markers
    code_block_patterns = [
        r"```python\n(.*?)```",  # Python-specific code blocks
        r"```\n(.*?)```",        # Generic code blocks
        r"`{3,}(.*?)`{3,}"      # Any code blocks with 3 or more backticks
    ]
    
    for pattern in code_block_patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            code = match.group(1).strip()
            if code:
                return code, True
    
    # If no code blocks found, try to find Python-like code directly
    lines = text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        # Skip empty lines at the start
        if not in_code and not line.strip():
            continue
            
        # Look for common Python code indicators
        if not in_code and (
            line.strip().startswith('def ') or
            line.strip().startswith('class ') or
            line.strip().startswith('import ') or
            line.strip().startswith('from ') or
            ':' in line
        ):
            in_code = True
            
        if in_code:
            code_lines.append(line)
            
    if code_lines:
        return '\n'.join(code_lines), True
        
    return "", False

def extract_mmlu_pro_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1), True
    else:
        return extract_mmlu_pro_answer_again(text)


def extract_mmlu_pro_answer_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1), True
    else:
        return extract_mmlu_pro_answer_final(text)


def extract_mmlu_pro_answer_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0), True
    else:
        print("answer extract failed\n")
        return None, False

def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df

def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res

def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(val_df, curr, k):
    prompt = ""
    with open(f"script/evaluate/eval_configs/mmlu-pro_initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["category"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt

def prepare_prompt(line: dict[str, Any]) -> str:
    query = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    query += f"Question: {line['question_content']}\n\n"
    if starter_code := line.get("starter_code", None):
        query += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
        query += f"```python\n{starter_code}\n```\n\n"
    else:
        query += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."
        query += "```python\n# YOUR CODE HERE\n```\n\n"
    return query


def lcb_codegeneration_prompt_fn(line):
    # For the prompt we need a more general function that can be used tweaked like in:
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    query = prepare_prompt(line)
    # List of dicts of the form: [{"input": "6\nabc\nacb\nbac\nbca\ncab\ncba\n", "output": "YES\nYES\nYES\nNO\nNO\nYES\n", "testtype": "stdin"}]
    public_test_cases = json.loads(line["public_test_cases"])
    private_test_cases = translate_private_test_cases(line["private_test_cases"])
    inputs = [test["input"] for test in public_test_cases + private_test_cases]
    outputs = [test["output"] for test in public_test_cases + private_test_cases]
    
    return query, inputs, outputs

def translate_private_test_cases(encoded_data: str) -> dict[str, str]:
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    return json.loads(original_data)

def prepare_multiple_choice_prompt(line: dict[str, Any], format_config: dict[str, Any]) -> tuple[str, str]:

    options_fields = format_config.get("options_fields", [])
    if len(options_fields) >= 4:  # Need at least 4 options for A, B, C, D
        # Get the options in a consistent order
        options = [line[field] for field in options_fields]

        correct_index = 0 # Not using randomized option

        """# Shuffle the options to randomize the correct answer position
        # Create a mapping from original positions to shuffled positions
        indices = list(range(len(options)))
        np.random.shuffle(indices)

        shuffled_options = [options[i] for i in indices]

        # Find where the correct answer ended up
        correct_index = indices.index(0)  # Assuming the first option is the correct one"""
        correct_letter = chr(65 + correct_index)  # A, B, C, D...

        options = {
            "A": options[0], # shuffled_options[0],
            "B": options[1], # shuffled_options[1],
            "C": options[2], # shuffled_options[2],
            "D": options[3] # shuffled_options[3]
        }
        answer = correct_letter

        # Format the problem with options
        formatted_problem = QUERY_TEMPLATE_MULTICHOICE.format(
            Question=line[format_config["question_field"]],
            A=options["A"],
            B=options["B"],
            C=options["C"],
            D=options["D"]
        )

        return formatted_problem, answer


# ===== Test functions to verify RouteLRM integration =====

def test_answer_extraction_and_comparison():
    """Test the improved answer extraction and comparison with RouteLRM methods."""
    print("Testing RouteLRM-based answer extraction and comparison...")
    print("=" * 70)

    # Test case 1: LaTeX fraction like \frac{200}{1}
    test_cases = [
        {
            "name": "LaTeX fraction \\frac{200}{1}",
            "text": "The answer is \\boxed{\\frac{200}{1}}",
            "expected_answer": "200",
            "should_match": True
        },
        {
            "name": "LaTeX fraction \\frac{100}{2}",
            "text": "The answer is \\boxed{\\frac{100}{2}}",
            "expected_answer": "50",
            "should_match": True
        },
        {
            "name": "Simple integer in boxed",
            "text": "Therefore, the answer is \\boxed{42}",
            "expected_answer": "42",
            "should_match": True
        },
        {
            "name": "LaTeX with sqrt",
            "text": "The result is \\boxed{\\sqrt{4}}",
            "expected_answer": "2",
            "should_match": True
        },
        {
            "name": "Nested braces",
            "text": "Answer: \\boxed{\\frac{\\sqrt{16}}{2}}",
            "expected_answer": "2",
            "should_match": True
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"Text: {test['text']}")

        # Extract answer
        extracted, has_answer = extract_boxed_answer(test['text'])
        print(f"Extracted: '{extracted}' (has_answer={has_answer})")

        # Check correctness
        if has_answer:
            is_correct = check_answer_correctness(extracted, test['expected_answer'], "boxed")
            print(f"Expected: '{test['expected_answer']}'")
            print(f"Match: {is_correct} (Expected: {test['should_match']})")

            if is_correct == test['should_match']:
                print("✓ PASS")
            else:
                print("✗ FAIL")
        else:
            print("✗ FAIL - No answer extracted")

    print("\n" + "=" * 70)
    print("Testing complete!")


if __name__ == "__main__":
    test_answer_extraction_and_comparison()
