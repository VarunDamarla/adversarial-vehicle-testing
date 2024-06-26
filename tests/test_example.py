from adversarial_vehicle_testing.example import double, triple


def test_double() -> None:
    assert double(2) == 4


def test_triple() -> None:
    assert triple(3) == 9
