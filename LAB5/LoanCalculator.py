from langchain_core.tools import tool


@tool
def calculate_loan_emi(
    principal: float, annual_interest_rate: float, tenure_months: int
) -> str:
    """Calculate the monthly EMI for a loan."""
    if principal <= 0:
        raise ValueError("Principal must be greater than 0.")
    if annual_interest_rate < 0:
        raise ValueError("Interest rate cannot be negative.")
    if tenure_months <= 0:
        raise ValueError("Tenure must be greater than 0 months.")

    monthly_rate = annual_interest_rate / 100 / 12
    if monthly_rate == 0:
        emi = principal / tenure_months
    else:
        emi = (
            principal
            * monthly_rate
            * (1 + monthly_rate) ** tenure_months
            / ((1 + monthly_rate) ** tenure_months - 1)
        )

    return f"Monthly EMI: {emi:.2f}"


def main() -> None:
    principal = float(input("Loan amount: "))
    annual_interest_rate = float(input("Annual interest rate (%): "))
    tenure_months = int(input("Tenure (months): "))

    print(calculate_loan_emi.invoke({
        "principal": principal,
        "annual_interest_rate": annual_interest_rate,
        "tenure_months": tenure_months,
    }))


if __name__ == "__main__":
    main()