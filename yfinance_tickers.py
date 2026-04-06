# yfinance_tickers.py

# -----------------------------
# CAC 40 (Euronext Paris .PA)
# -----------------------------
CAC40 = {
    "Accor": "AC.PA",
    "Air Liquide": "AI.PA",
    "Airbus": "AIR.PA",
    "ArcelorMittal": "MT.PA",
    "Atos": "ATO.PA",
    "AXA": "CS.PA",
    "BNP Paribas": "BNP.PA",
    "Bouygues": "EN.PA",
    "Bureau Veritas": "BVI.PA",
    "Capgemini": "CAP.PA",
    "Carrefour": "CA.PA",
    "Crédit Agricole": "ACA.PA",
    "Danone": "BN.PA",
    "Dassault Systèmes": "DSY.PA",
    "Engie": "ENGI.PA",
    "EssilorLuxottica": "EL.PA",
    "Hermès International": "RMS.PA",
    "Kering": "KER.PA",
    "L'Oréal": "OR.PA",
    "Legrand": "LR.PA",
    "LVMH": "MC.PA",
    "Michelin": "ML.PA",
    "Orange": "ORA.PA",
    "Pernod Ricard": "RI.PA",
    "Publicis Groupe": "PUB.PA",
    "Renault": "RNO.PA",
    "Safran": "SAF.PA",
    "Saint‑Gobain": "SGO.PA",
    "Sanofi": "SAN.PA",
    "Schneider Electric": "SU.PA",
    "Société Générale": "GLE.PA",
    "STMicroelectronics": "STM.PA",
    "Teleperformance": "TEP.PA",
    "Thales": "HO.PA",
    "TotalEnergies": "TTE.PA",
    "Unibail‑Rodamco‑Westfield": "URW.PA",
    "Veolia Environnement": "VIE.PA",
    "Vinci": "DG.PA",
    "Vivendi": "VIV.PA",
    "Worldline": "WLN.PA"
}

# -----------------------------
# US Large Cap (NASDAQ/NYSE)
# -----------------------------
US_LARGE_CAP = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Alphabet Class A": "GOOGL",
    "Alphabet Class C": "GOOG",
    "Tesla": "TSLA",
    "Meta Platforms": "META",
    "NVIDIA": "NVDA",
    "JPMorgan Chase": "JPM",
    "Johnson & Johnson": "JNJ",
    "Visa": "V",
    "Mastercard": "MA",
    "Walmart": "WMT",
    "Procter & Gamble": "PG",
    "Home Depot": "HD",
    "Bank of America": "BAC",
    "Wells Fargo": "WFC",
    "Pfizer": "PFE",
    "Merck": "MRK",
    "Coca‑Cola": "KO",
    "PepsiCo": "PEP",
    "Intel": "INTC",
    "Cisco Systems": "CSCO",
    "Adobe": "ADBE",
    "Salesforce": "CRM",
    "Oracle": "ORCL",
    "Netflix": "NFLX",
    "PayPal": "PYPL"
}

# -----------------------------
# Major ETFs (US)
# -----------------------------
US_ETFS = {
    "SPDR S&P 500 ETF Trust": "SPY",
    "Invesco QQQ Trust": "QQQ",
    "Vanguard S&P 500 ETF": "VOO",
    "iShares Russell 2000 ETF": "IWM",
    "Vanguard Total Stock Market ETF": "VTI",
    "iShares MSCI Emerging Markets ETF": "EEM",
    "iShares 20+ Year Treasury Bond ETF": "TLT",
    "iShares Gold Trust": "IAU",
    "SPDR Gold Shares": "GLD"
}

# -----------------------------
# UK & Europe
# -----------------------------
EUROPE = {
    "Unilever PLC": "ULVR.L",
    "HSBC Holdings": "HSBA.L",
    "BP PLC": "BP.L",
    "Royal Dutch Shell A": "RDSA.L",
    "Diageo PLC": "DGE.L",
    "Rio Tinto": "RIO.L",
    "British American Tobacco": "BATS.L",
    "GlaxoSmithKline": "GSK.L",
    "Volkswagen AG": "VOW3.DE",
    "Siemens AG": "SIE.DE",
    "SAP SE": "SAP.DE"
}

# -----------------------------
# Major Cryptocurrencies (yfinance format)
# -----------------------------
CRYPTO = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Cardano": "ADA-USD",
    "XRP": "XRP-USD",
    "Dogecoin": "DOGE-USD"
}

# -----------------------------
# Market Index Symbols
# -----------------------------
INDICES = {
    "S&P 500 Index": "^GSPC",
    "Dow Jones Industrial Average": "^DJI",
    "NASDAQ Composite": "^IXIC",
    "CAC 40 Index": "^FCHI",
    "FTSE 100 Index": "^FTSE",
    "DAX Index": "^GDAXI"
}

# -----------------------------
# Combined dictionary
# -----------------------------
TICKERS_DICT = {
    **CAC40,
    **US_LARGE_CAP,
    **US_ETFS,
    **EUROPE,
    **CRYPTO,
    **INDICES
}

# List of names for search UI
TICKER_NAMES = list(TICKERS_DICT.keys())
