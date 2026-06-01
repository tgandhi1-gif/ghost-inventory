"""The Ghost Inventory - reconcile ERP, scanner and sales to find phantom SKUs."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
N = 310_000

items = pd.DataFrame({
    "sku": [f"SKU{i:06d}" for i in range(N)],
    "unit_cost": rng.gamma(2.0, 8.0, N).round(2),
    "reorder_qty": rng.poisson(4, N),
})

scanned = rng.random(N) < 0.84
sold = rng.random(N) < 0.80

items["ghost"] = (~scanned) & (~sold) & (items["reorder_qty"] > 0)
ghosts = items[items["ghost"]].copy()
ghosts["cash_tied"] = (ghosts["unit_cost"] * ghosts["reorder_qty"] * 4).round(2)

print(f"Ghost SKUs found: {len(ghosts):,}")
print(f"Cash tied up in ghost reorders: {ghosts['cash_tied'].sum():,.0f} USD")

top = ghosts.sort_values("cash_tied", ascending=False).head(15)
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(9, 5))
plt.barh(top["sku"], top["cash_tied"], color="#ff6a3d")
plt.gca().invert_yaxis()
plt.xlabel("Cash tied up (USD)")
plt.title("Top ghost SKUs draining the reorder budget")
plt.tight_layout()
plt.savefig("outputs/ghost_inventory.png", dpi=120)
print("Saved outputs/ghost_inventory.png")
