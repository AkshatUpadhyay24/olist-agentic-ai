from contextlib import redirect_stdout
from io import StringIO

from rag.rag_tool import search_knowledge


TEST_CASES = [

    {
        "question": "What is the standard delivery time for an order?",
        "expected_section": "DELIVERY POLICY"
    },

    {
        "question": "How long does a normal shipment usually take to reach the customer?",
        "expected_section": "DELIVERY POLICY"
    },

    {
        "question": "Can I cancel an order after it has already been shipped?",
        "expected_section": "ORDER POLICY"
    },

    {
        "question": "Is cancellation possible once the seller has shipped the order?",
        "expected_section": "ORDER POLICY"
    },

    {
        "question": "What kinds of products can generally be returned?",
        "expected_section": "RETURN POLICY"
    },

    {
        "question": "If an item is damaged or defective, can it be sent back?",
        "expected_section": "RETURN POLICY"
    },

    {
        "question": "When are refunds generally processed?",
        "expected_section": "REFUND POLICY"
    },

    {
        "question": "What happens before a refund is processed?",
        "expected_section": "REFUND POLICY"
    },

    {
        "question": "What should a customer provide when contacting support about an order?",
        "expected_section": "CUSTOMER SUPPORT"
    },

    {
        "question": "What identifier should I give support for an order issue?",
        "expected_section": "CUSTOMER SUPPORT"
    },

    {
        "question": "What is the company's warranty policy for products?",
        "expected_section": None
    }
]


def evaluate_case(test_case):

    question = test_case["question"]
    expected_section = test_case["expected_section"]

    # Hide detailed RAG logs so evaluation output stays clean
    captured_output = StringIO()

    with redirect_stdout(captured_output):

        result = search_knowledge(
            question,
            top_k=3
        )

    found = result.get("found", False)

    results = result.get(
        "results",
        []
    )

    retrieved_sections = [
        item["section"]
        for item in results
    ]

    top_section = None

    if retrieved_sections:
        top_section = retrieved_sections[0]

    # --------------------------------------------------
    # CASE 1: We expect a relevant section
    # --------------------------------------------------

    if expected_section is not None:

        hit_at_1 = (
            top_section == expected_section
        )

        hit_at_3 = (
            expected_section in retrieved_sections
        )

        passed = hit_at_1

        return {
            "question": question,
            "expected": expected_section,
            "retrieved": retrieved_sections,
            "top_section": top_section,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "false_positive": False,
            "passed": passed
        }

    # --------------------------------------------------
    # CASE 2: We expect NO relevant information
    # --------------------------------------------------

    false_positive = found

    passed = not false_positive

    return {
        "question": question,
        "expected": "NONE",
        "retrieved": retrieved_sections,
        "top_section": top_section,
        "hit_at_1": False,
        "hit_at_3": False,
        "false_positive": false_positive,
        "passed": passed
    }


def main():

    print("\n")
    print("=" * 80)
    print("RAG V5 - RETRIEVAL EVALUATION")
    print("=" * 80)

    total = len(TEST_CASES)

    hit_at_1_count = 0
    hit_at_3_count = 0
    false_positive_count = 0
    passed_count = 0

    results = []

    for index, test_case in enumerate(
        TEST_CASES,
        start=1
    ):

        result = evaluate_case(
            test_case
        )

        results.append(result)

        if result["hit_at_1"]:
            hit_at_1_count += 1

        if result["hit_at_3"]:
            hit_at_3_count += 1

        if result["false_positive"]:
            false_positive_count += 1

        if result["passed"]:
            passed_count += 1

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print("\n" + "-" * 80)

        print(
            f"Test {index}: {status}"
        )

        print(
            f"Question : {result['question']}"
        )

        print(
            f"Expected : {result['expected']}"
        )

        print(
            f"Retrieved: {result['retrieved']}"
        )

        print(
            f"Hit@1   : {result['hit_at_1']}"
        )

        print(
            f"Hit@3   : {result['hit_at_3']}"
        )

        print(
            f"False + : {result['false_positive']}"
        )

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    normal_cases = sum(
        1
        for test in TEST_CASES
        if test["expected_section"] is not None
    )

    hit_at_1 = (
        hit_at_1_count / normal_cases
        if normal_cases
        else 0
    )

    hit_at_3 = (
        hit_at_3_count / normal_cases
        if normal_cases
        else 0
    )

    pass_rate = (
        passed_count / total
        if total
        else 0
    )

    print("\n")
    print("=" * 80)
    print("RAG EVALUATION SUMMARY")
    print("=" * 80)

    print(
        f"Total test cases       : {total}"
    )

    print(
        f"Passed cases           : {passed_count}"
    )

    print(
        f"Failed cases           : "
        f"{total - passed_count}"
    )

    print(
        f"Hit@1                  : "
        f"{hit_at_1:.2%}"
    )

    print(
        f"Hit@3                  : "
        f"{hit_at_3:.2%}"
    )

    print(
        f"False positives        : "
        f"{false_positive_count}"
    )

    print(
        f"Overall pass rate      : "
        f"{pass_rate:.2%}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()