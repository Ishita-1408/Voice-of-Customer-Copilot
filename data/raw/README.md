# data/raw/README.md

# Raw Public Data Provenance & Attribution

This directory contains provenance documentation for the public customer review dataset utilized by the Voice of Customer Copilot MVP.

## 1. Source & Attribution
- **Dataset**: Public E-Commerce Product Customer Reviews (Amazon Review Data).
- **Public Domain**: Sourced from public product reviews under standard academic and benchmarking terms.
- **Component Size**: 350 genuine customer review records.

## 2. Dataset Composition
The canonical dataset (`data/processed/voc_feedback.csv`, 800 records total) combines:
1. **350 Genuine Public Product Reviews**: Raw customer feedback spanning product quality, physical durability, shipping experiences, and general product advocacy.
2. **450 Clearly Labelled Synthetic Demonstration Records**: Generated to represent multi-channel customer interactions across user interviews, customer support tickets, surveys, and mobile app store feedback.

## 3. Provenance & Privacy Notice
- **No Private or Internal Company Data**: All public reviews are open-domain e-commerce feedback.
- **Synthetic Differentiation**: Synthetic records are labelled with `metadata_origin: "synthetic_demonstration"`. They represent demonstration scenarios for Voice-of-Customer analytics evaluation and do not represent actual internal company records.
