# Results Figures

Clean, essential visualizations for the organoid DILI prediction analysis.

## Directory Structure

### `core/` - Key Results
- `key_results_summary.png` - Main performance comparison
- `event_aware_final_summary.png` - Event-aware feature results (if available)
- `media_change_events_summary.png` - Media change event overview (if available)

### `phase2/` - Phase 2 Embeddings
- Hierarchical clustering visualizations for each embedding method
- Shows wells → concentrations → drugs progression

### `validation/` - Event Verification  
- `plate_*_event_verification.png` - Time series with detected events
- `spike_characterization_detailed.png` - Detailed spike analysis
- `event_timing_summary_all_plates.png` - Event timing overview

## Regeneration

To regenerate all figures:

```bash
python regenerate_figures.py
```

Individual modules can also be run directly from `src/` directory.
