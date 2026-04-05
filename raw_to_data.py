import pandas as pd
import numpy as np
import yfinance as yf


def main():
    # 8 stocks CAC40
    # Air Liquide, AXA, BNP Paribas, Carrefour, Crédit Agricole, LVMH, Safran, TotalEnergies
    #syms = sorted(['AI.PA', 'CS.PA', 'BNP.PA', 'CA.PA','ACA.PA', 'MC.PA', 'SAF.PA', 'TTE.PA'])
    syms = sorted(['AI.PA', 'MC.PA', 'BNP.PA', 'CA.PA', 'ACA.PA', 'CS.PA', 'OR.PA', 
    'RNO.PA', 'ENGI.PA', 'KER.PA', 'SAN.PA', 'CAP.PA', 'DG.PA', 'VIE.PA', 'SU.PA', 
    'RI.PA', 'GLE.PA', 'MT.AS'])
    n_dims = len(syms)
    fit = True #?

    data = yf.download(syms, start='2010-01-01', end='2025-12-30',auto_adjust=True)["Close"]
    y = np.log(data).diff().dropna()
    np.savez_compressed("data.npz", n_dims=n_dims,data=data)
    
if __name__ == "__main__":
    main()