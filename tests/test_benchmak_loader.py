from app.benchmarks.loader import BenchmarkLoader


def test_benchmark_loading():

    loader = BenchmarkLoader()

    cases = loader.load()

    assert len(cases) >= 50


def test_benchmark_categories():
    loader = BenchmarkLoader()
    cases = loader.load()

    categories = {case.category for case in cases}

    expected_categories = {
        "normal",
        "direct_injection",
        "indirect_injection",
        "unsafe_tool",
        "out_of_scope"
    }

    assert categories == expected_categories


def test_benchmark_data_types():

    loader = BenchmarkLoader()

    cases = loader.load()

    for case in cases:

        assert isinstance(case.test_id, str)
        assert isinstance(case.task, str)
        assert case.expected_status in [
            "success",
            "failure",
            "refused"
        ]
