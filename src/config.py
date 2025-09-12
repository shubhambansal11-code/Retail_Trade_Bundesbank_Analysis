# src/config.py

BUNDESBANK_URLS = {
    "Electronics": "https://api.statistiken.bundesbank.de/rest/download/BBDE1/M.DE.Y.GUA1.N2G470000.A.V.I15.A?format=csv&lang=en",
    "Groceries": "https://api.statistiken.bundesbank.de/rest/download/BBDE1/M.DE.Y.GUA1.P2XG71020.A.V.I15.A?format=csv&lang=en",
    "Textiles": "https://api.statistiken.bundesbank.de/rest/download/BBDE1/M.DE.Y.GUA1.P2XG75060.A.V.I15.A?format=csv&lang=en",
    "Furniture": "https://api.statistiken.bundesbank.de/rest/download/BBDE1/M.DE.Y.GUA1.P2XG75064.A.V.I15.A?format=csv&lang=en",
    "Pharmacy": "https://api.statistiken.bundesbank.de/rest/download/BBDE1/M.DE.Y.GUA1.P2XG75070.A.V.I15.A?format=csv&lang=en",
    "Motor Vehicles": "https://api.statistiken.bundesbank.de/rest/download/BBDE1/M.DE.Y.GUA1.N2G450000.A.V.I15.A?format=csv&lang=en",
}

# Default revenue mapping, these can be changed need to be careful while putting this into a stream_lit version, as of now they are not given as inputs
BASE_REVENUE_LATEST = {
    "Electronics": 5_000_000,
    "Groceries": 12_000_000,
    "Textiles": 2_000_000,
    "Furniture": 1_500_000,
    "Pharmacy": 3_000_000,
    "Motor Vehicles": 8_000_000,
}

# Inventory parameters
INVENTORY_THRESHOLD = 0.40
SHORTAGE_ALPHA = 1.0