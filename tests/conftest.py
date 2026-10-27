import pytest

BASE_URL = "http://localhost:8000"

@pytest.fixture
def base_url():
    return BASE_URL

@pytest.fixture
def legit_tx():
    """Low-risk transaction — expected fraud_probability well below threshold."""
    return {
        "TransactionAmt": 50.0,
        "TransactionDT": 86400,
        "card1": 9500.0, "card2": 325.0,
        "card4": "visa", "card6": "debit",
        "addr1": 315.0,
        "P_emaildomain": "gmail.com", "R_emaildomain": "gmail.com",
        "ProductCD": "W",
        "M1": "T", "M2": "T", "M3": "T", "M4": "M0",
        "M5": "F", "M6": "T", "M7": "F", "M8": "F", "M9": "F",
    }

@pytest.fixture
def high_risk_tx():
    """High-amount transaction with suspicious signals."""
    return {
        "TransactionAmt": 9999.0,
        "TransactionDT": 7200,
        "card1": 1.0, "card2": 1.0,
        "card4": "mastercard", "card6": "credit",
        "addr1": 1.0,
        "P_emaildomain": "anonymous.com", "R_emaildomain": "anonymous.com",
        "ProductCD": "C",
        "M1": "F", "M2": "F", "M3": "F", "M4": "M2",
        "M5": "T", "M6": "F", "M7": "T", "M8": "T", "M9": "T",
    }
