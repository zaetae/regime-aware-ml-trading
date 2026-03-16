"""Measurement script — run after every detector change to track event rates."""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.load_data import load_spy
from src.patterns.support_resistance import calculate_support_resistance
from src.patterns.triangles import detect_triangle_pattern
from src.patterns.channels import detect_channel
from src.patterns.multiple_tops_bottoms import detect_multiple_tops_bottoms

df = load_spy()

sr_df = calculate_support_resistance(df)
sr = sr_df["near_support"] | sr_df["near_resistance"]

tri_df = detect_triangle_pattern(df)
tri = tri_df["triangle_pattern"].notna()

ch_df = detect_channel(df)
ch = ch_df["channel_pattern"].notna()

mtb_df = detect_multiple_tops_bottoms(df)
mtb = mtb_df["multiple_top_bottom_pattern"].notna()

combined = sr | tri | ch | mtb

n = len(df)
print(f"{'Detector':<25} {'Count':>6}  {'Rate':>7}")
print("-" * 42)
print(f"{'Support/Resistance':<25} {sr.sum():>6}  {sr.mean()*100:>6.1f}%")
print(f"{'Triangles':<25} {tri.sum():>6}  {tri.mean()*100:>6.1f}%")
print(f"{'Channels':<25} {ch.sum():>6}  {ch.mean()*100:>6.1f}%")
print(f"{'Multi Top/Bottom':<25} {mtb.sum():>6}  {mtb.mean()*100:>6.1f}%")
print("-" * 42)
print(f"{'COMBINED (any event)':<25} {combined.sum():>6}  {combined.mean()*100:>6.1f}%")
print(f"\nTarget: 400-1,200 combined events ({400/n*100:.1f}%-{1200/n*100:.1f}%)")
