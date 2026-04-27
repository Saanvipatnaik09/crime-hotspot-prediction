def risk_statement(prob, location="Selected Area"):
    if prob >= 0.75:
        level = "HIGH"
    elif prob >= 0.4:
        level = "MEDIUM"
    else:
        level = "LOW"

    return f"""
📍 Location: {location}

🚨 Risk Level: {level}
📊 Risk Probability: {prob:.2f}

🧠 Interpretation:
Based on historical spatial and temporal crime patterns,
this area shows a **{level.lower()} likelihood** of crime occurrence.
Predictions are probabilistic, not deterministic.
"""
