# The payload matches the 'predict_fn' in your inference.py
payload = {
    "inputs": "Transaction Details: Customer: suspicious_user@email.com, IP Location: Russia, Shipping Address: Florida, USA, Items: 5x $500 Apple Gift Cards, Previous Orders: 0, Account Age: 15 minutes.",
    "parameters": {
        "max_new_tokens": 100,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95
    }
}

# Invoke the endpoint
response = predictor.predict(payload)

# Print the response from the API
print(response)