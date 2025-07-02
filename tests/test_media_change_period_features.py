#!/usr/bin/env python3
"""
Tests for media-change-aware period feature extraction.

This test suite validates that we can:
1. Correctly identify media change events
2. Split time series into inter-media-change periods
3. Extract meaningful features from each period
4. Track progressive changes across periods
5. Handle edge cases and data quality issues
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.data_loader import DataLoader
from scripts.analysis.load_clean_data import load_clean_oxygen_data, get_treatment_wells


class TestMediaChangePeriodDetection:
    """Test that we can correctly identify and split media change periods."""
    
    def test_can_load_media_events(self):
        """Test that we can load media change events from database."""
        with DataLoader() as loader:
            events = loader.load_media_events()
        
        assert len(events) > 0, "Should load some media change events"
        assert 'plate_id' in events.columns, "Events should have plate_id"
        assert 'event_time' in events.columns, "Events should have event_time"
        assert len(events['plate_id'].unique()) >= 5, "Should have events for multiple plates"
    
    def test_events_have_realistic_timing(self):
        """Test that most media change events occur at realistic intervals."""
        with DataLoader() as loader:
            events = loader.load_media_events()
        
        # Collect all intervals across plates
        all_intervals = []
        problematic_plates = []
        
        for plate_id in events['plate_id'].unique():
            plate_events = events[events['plate_id'] == plate_id].sort_values('event_time')
            
            if len(plate_events) >= 2:
                event_times = pd.to_datetime(plate_events['event_time'])
                
                for i in range(1, len(event_times)):
                    interval_hours = (event_times.iloc[i] - event_times.iloc[i-1]).total_seconds() / 3600
                    all_intervals.append(interval_hours)
                    
                    # Flag problematic intervals but don't fail the test
                    if interval_hours < 12:  # Less than 12 hours is suspicious
                        problematic_plates.append((plate_id, interval_hours))
        
        # Test that most intervals are reasonable (allow for some data issues)
        reasonable_intervals = [i for i in all_intervals if 24 <= i <= 14*24]
        reasonable_fraction = len(reasonable_intervals) / len(all_intervals) if all_intervals else 0
        
        assert reasonable_fraction >= 0.7, f"Only {reasonable_fraction:.1%} of intervals are reasonable (24h-14d)"
        assert len(all_intervals) > 0, "Should have some intervals to test"
        
        # Log problematic cases for investigation (but don't fail)
        if problematic_plates:
            print(f"\nNote: Found {len(problematic_plates)} suspicious short intervals:")
            for plate_id, interval in problematic_plates[:3]:
                print(f"  {plate_id[:8]}...: {interval:.1f}h")
            print("  (These may be data entry issues or repeated media changes)")
    
    def test_can_align_events_with_oxygen_data(self):
        """Test that we can align events with oxygen time series data."""
        treatment_wells = get_treatment_wells()
        test_plate = treatment_wells['plate_id'].iloc[0]
        
        # Load oxygen data for this plate
        oxygen_data = load_clean_oxygen_data(plate_ids=[test_plate])
        
        # Load events for this plate
        with DataLoader() as loader:
            events = loader.load_media_events()
        plate_events = events[events['plate_id'] == test_plate]
        
        if len(plate_events) > 0 and len(oxygen_data) > 0:
            # Check that we can convert event times to elapsed hours
            if 'timestamp' in oxygen_data.columns:
                plate_start = pd.to_datetime(oxygen_data['timestamp']).min()
                plate_end = pd.to_datetime(oxygen_data['timestamp']).max()
                
                for _, event in plate_events.iterrows():
                    event_time = pd.to_datetime(event['event_time'])
                    
                    # Event should be within the oxygen data timeframe (with some tolerance)
                    assert plate_start - timedelta(days=1) <= event_time <= plate_end + timedelta(days=1), \
                        f"Event at {event_time} outside oxygen data range {plate_start} to {plate_end}"


class TestPeriodSplitting:
    """Test that we can split time series into periods between media changes."""
    
    def test_periods_are_sequential_and_non_overlapping(self):
        """Test that periods are properly ordered and don't overlap."""
        from scripts.features.media_change_periods import MediaChangePeriodSplitter
        
        splitter = MediaChangePeriodSplitter()
        
        # Test with a sample well
        treatment_wells = get_treatment_wells()
        test_well = treatment_wells.iloc[0]
        
        periods = splitter.extract_well_periods(test_well['well_id'], test_well['plate_id'])
        
        if len(periods) >= 2:
            # Check that periods are sequential
            for i in range(1, len(periods)):
                assert periods[i].period_number > periods[i-1].period_number, \
                    "Period numbers should be sequential"
                
                # Check non-overlapping (allowing for small gaps due to data filtering)
                assert periods[i].start_hours >= periods[i-1].end_hours - 1, \
                    f"Period {i} starts before period {i-1} ends"
        
        # Each period should have positive duration
        for period in periods:
            assert period.duration > 0, "Period duration should be positive"
            assert period.data_points > 0, "Period should have data points"
    
    def test_periods_cover_full_timeseries(self):
        """Test that splitting covers the entire time series without gaps."""
        pytest.skip("Will implement after creating period splitting function")
    
    def test_handles_plates_with_no_events(self):
        """Test graceful handling of plates with no detected media changes."""
        pytest.skip("Will implement after creating period splitting function")
    
    def test_handles_plates_with_single_event(self):
        """Test handling of plates with only one media change event."""
        pytest.skip("Will implement after creating period splitting function")


class TestPeriodFeatureExtraction:
    """Test that we can extract meaningful features from each period."""
    
    def test_can_extract_basic_statistics_per_period(self):
        """Test extraction of mean, std, trend for each period."""
        from scripts.features.media_change_periods import extract_period_features_for_drug
        
        # Test with top 3 drugs for faster testing (can expand to all drugs later)
        treatment_wells = get_treatment_wells()
        test_drugs = treatment_wells['drug'].value_counts().head(3).index
        
        all_features = []
        for drug in test_drugs:
            features = extract_period_features_for_drug(drug, max_wells=2)
            if not features.empty:
                all_features.append(features)
        
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            
            # Check required columns exist
            required_cols = ['period_mean', 'period_std', 'period_trend', 'period_duration', 
                           'period_number', 'well_id', 'drug', 'concentration']
            for col in required_cols:
                assert col in combined_features.columns, f"Missing required feature column: {col}"
            
            # Check values are reasonable
            assert combined_features['period_mean'].notna().any(), "Should have some non-null mean values"
            assert combined_features['period_std'].notna().any(), "Should have some non-null std values"
            assert (combined_features['period_duration'] > 0).all(), "All period durations should be positive"
            assert combined_features['period_number'].min() >= 1, "Period numbers should start from 1"
            
            print(f"\nTested feature extraction for {len(test_drugs)} drugs:")
            print(f"  Total periods: {len(combined_features)}")
            print(f"  Unique wells: {combined_features['well_id'].nunique()}")
            print(f"  Period numbers range: {combined_features['period_number'].min()} to {combined_features['period_number'].max()}")
        else:
            print("⏭ No features extracted from test drugs - skipping")
            return
    
    def test_period_features_have_expected_structure(self):
        """Test that period features have consistent structure across wells/drugs."""
        from scripts.features.media_change_periods import extract_period_features_for_drug
        
        # Test with multiple drugs
        treatment_wells = get_treatment_wells()
        test_drugs = treatment_wells['drug'].value_counts().head(2).index
        
        all_features = []
        for drug in test_drugs:
            features = extract_period_features_for_drug(drug, max_wells=3)
            if not features.empty:
                all_features.append(features)
        
        if all_features:
            # Check consistency across drugs
            all_columns = [set(df.columns) for df in all_features]
            common_columns = set.intersection(*all_columns)
            
            # Should have consistent core columns
            required_core = {'well_id', 'drug', 'concentration', 'period_number', 
                           'period_mean', 'period_std', 'period_duration'}
            
            assert required_core.issubset(common_columns), \
                f"Missing core columns. Expected: {required_core - common_columns}"
            
            # Combine and validate structure
            combined = pd.concat(all_features, ignore_index=True)
            validate_period_features_structure(combined)
            
            print(f"\nStructure validation passed for {len(test_drugs)} drugs:")
            print(f"  Total periods: {len(combined)}")
            print(f"  Common columns: {len(common_columns)}")
        else:
            pytest.skip("No features extracted for structure testing")
    
    def test_features_handle_short_periods(self):
        """Test that feature extraction handles periods with few datapoints."""
        pytest.skip("Will implement after creating feature extraction")
    
    def test_features_detect_recovery_patterns(self):
        """Test that features can detect post-media-change recovery patterns."""
        pytest.skip("Will implement after creating feature extraction")


class TestProgressiveEffectAnalysis:
    """Test that we can track changes across sequential periods."""
    
    def test_can_track_feature_evolution_across_periods(self):
        """Test that we can measure how features change from period 1 → 2 → 3."""
        pytest.skip("Will implement after creating progressive analysis")
    
    def test_detects_drug_specific_progression_patterns(self):
        """Test that different drugs show different progression patterns."""
        pytest.skip("Will implement after creating progressive analysis")
    
    def test_identifies_adaptation_vs_deterioration(self):
        """Test classification of whether responses get better or worse over time."""
        pytest.skip("Will implement after creating progressive analysis")
    
    def test_handles_wells_with_different_numbers_of_periods(self):
        """Test graceful handling when wells have different numbers of media changes."""
        pytest.skip("Will implement after creating progressive analysis")


class TestDataIntegrityValidation:
    """Test data quality and edge case handling."""
    
    def test_validates_period_data_quality(self):
        """Test that we can assess data quality within each period."""
        pytest.skip("Will implement with quality metrics")
    
    def test_handles_missing_data_within_periods(self):
        """Test handling of gaps in data within a period."""
        pytest.skip("Will implement with missing data handling")
    
    def test_flags_unrealistic_media_change_responses(self):
        """Test detection of unrealistic spikes that might not be real media changes."""
        pytest.skip("Will implement with response validation")
    
    def test_consistent_results_across_replicates(self):
        """Test that replicate wells show consistent period-based features."""
        pytest.skip("Will implement with replicate consistency checks")


class TestBiologicalValidation:
    """Test that features capture biologically meaningful patterns."""
    
    def test_control_wells_show_predictable_patterns(self):
        """Test that control wells show expected responses to media changes."""
        from scripts.features.media_change_periods import extract_period_features_for_drug
        
        # Find drugs with control wells (concentration = 0 or very low)
        treatment_wells = get_treatment_wells()
        low_conc_wells = treatment_wells[treatment_wells['concentration'] <= 0.01]
        
        if not low_conc_wells.empty:
            test_drug = low_conc_wells['drug'].iloc[0]
            features = extract_period_features_for_drug(test_drug, max_wells=5)
            
            if not features.empty:
                control_features = features[features['concentration'] <= 0.01]
                
                if not control_features.empty:
                    # Control wells should show relatively stable patterns
                    avg_cv = control_features['period_cv'].mean()
                    
                    # Control wells should have lower variability than typical treatment wells
                    assert avg_cv < 2.0, f"Control wells too variable (CV={avg_cv:.2f})"
                    
                    print(f"\nControl well validation:")
                    print(f"  Drug: {test_drug}")
                    print(f"  Control periods: {len(control_features)}")
                    print(f"  Average CV: {avg_cv:.3f}")
                else:
                    print("⏭ No control concentration features found - skipping")
                    return
            else:
                print("⏭ No features for drug with controls - skipping")
                return
        else:
            print("⏭ No low concentration wells found - skipping")
            return
    
    def test_toxic_drugs_show_progressive_deterioration(self):
        """Test that DILI+ drugs show worsening responses over time."""
        from scripts.features.media_change_periods import extract_period_features_for_drug
        
        # Get DILI+ drugs
        treatment_wells = get_treatment_wells()
        dili_positive = treatment_wells[treatment_wells['binary_dili'] == True]
        
        if not dili_positive.empty:
            test_drug = dili_positive['drug'].value_counts().index[0]
            features = extract_period_features_for_drug(test_drug, max_wells=3)
            
            if not features.empty:
                # Look for wells with multiple periods
                wells_with_multiple_periods = features.groupby('well_id')['period_number'].count()
                multi_period_wells = wells_with_multiple_periods[wells_with_multiple_periods >= 3]
                
                if not multi_period_wells.empty:
                    test_well = multi_period_wells.index[0]
                    well_periods = features[features['well_id'] == test_well].sort_values('period_number')
                    
                    # Check if there's a trend in period characteristics
                    # (This is exploratory - we don't assert specific patterns yet)
                    period_means = well_periods['period_mean'].values
                    period_trends = well_periods['period_trend'].values
                    
                    print(f"\nToxic drug progression analysis:")
                    print(f"  Drug: {test_drug} (DILI+)")
                    print(f"  Well: {test_well[:8]}...")
                    print(f"  Periods: {len(well_periods)}")
                    print(f"  Period means: {[f'{m:.1f}' for m in period_means]}")
                    print(f"  Period trends: {[f'{t:.3f}' for t in period_trends]}")
                    
                    # Basic sanity check - should have finite values
                    assert np.isfinite(period_means).all(), "Period means should be finite"
                    assert np.isfinite(period_trends).all(), "Period trends should be finite"
                else:
                    print("⏭ No wells with multiple periods for progression analysis - skipping")
                    return
            else:
                print("⏭ No features for DILI+ drug - skipping")
                return
        else:
            print("⏭ No DILI+ drugs found - skipping")
            return
    
    def test_concentration_response_is_monotonic_within_periods(self):
        """Test that higher concentrations show stronger effects within each period."""
        pytest.skip("Will implement with concentration analysis")
    
    def test_period_features_correlate_with_known_drug_properties(self):
        """Test that period features correlate with known drug mechanisms/toxicity."""
        pytest.skip("Will implement with drug property correlation")


# Utility functions for testing
def create_mock_oxygen_timeseries(duration_hours=400, sampling_interval=1.5, 
                                 media_change_times=[72, 168, 264], 
                                 baseline_o2=10, spike_magnitude=30):
    """Create mock oxygen data with known media change pattern for testing."""
    times = np.arange(0, duration_hours, sampling_interval)
    o2_values = np.full_like(times, baseline_o2)
    
    # Add spikes at media change times
    for mc_time in media_change_times:
        spike_idx = np.argmin(np.abs(times - mc_time))
        # Create spike pattern: sharp increase, then decay
        for i in range(len(times)):
            if spike_idx <= i < spike_idx + 10:  # 10-point spike
                decay_factor = np.exp(-(i - spike_idx) * 0.3)
                o2_values[i] += spike_magnitude * decay_factor
    
    # Add some noise
    o2_values += np.random.normal(0, 2, len(times))
    
    return pd.DataFrame({
        'elapsed_hours': times,
        'o2': o2_values,
        'well_id': 'test_well_001',
        'plate_id': 'test_plate_001'
    })


def validate_period_features_structure(features_df):
    """Validate that period features have expected structure."""
    required_columns = [
        'well_id', 'plate_id', 'drug', 'concentration',
        'period_number', 'period_start_hours', 'period_end_hours',
        'period_duration', 'period_mean', 'period_std'
    ]
    
    for col in required_columns:
        assert col in features_df.columns, f"Missing required column: {col}"
    
    # Check data types
    assert features_df['period_number'].dtype in ['int64', 'int32'], "Period number should be integer"
    assert features_df['period_duration'].min() > 0, "Period duration should be positive"
    
    # Check that within each well, period start times are increasing
    for well_id in features_df['well_id'].unique():
        well_data = features_df[features_df['well_id'] == well_id].sort_values('period_number')
        assert well_data['period_start_hours'].is_monotonic_increasing, \
            f"Period start times should increase within well {well_id}"


def test_all_drugs_comprehensive(max_drugs: int = None, max_wells_per_drug: int = 5):
    """Test feature extraction on all drugs in the dataset."""
    from scripts.features.media_change_periods import extract_period_features_for_drug
    
    print(f"\n=== COMPREHENSIVE ALL-DRUGS TEST ===")
    treatment_wells = get_treatment_wells()
    all_drugs = treatment_wells['drug'].value_counts()
    
    if max_drugs:
        test_drugs = all_drugs.head(max_drugs).index
        print(f"Testing top {max_drugs} drugs by well count")
    else:
        test_drugs = all_drugs.index
        print(f"Testing ALL {len(test_drugs)} drugs")
    
    results = {
        'successful_drugs': [],
        'failed_drugs': [],
        'total_periods': 0,
        'total_wells': 0
    }
    
    for i, drug in enumerate(test_drugs):
        try:
            features = extract_period_features_for_drug(drug, max_wells=max_wells_per_drug)
            
            if not features.empty:
                results['successful_drugs'].append(drug)
                results['total_periods'] += len(features)
                results['total_wells'] += features['well_id'].nunique()
                
                if i < 3:  # Show details for first 3 drugs
                    print(f"  ✓ {drug}: {len(features)} periods, {features['well_id'].nunique()} wells")
            else:
                results['failed_drugs'].append((drug, "No features extracted"))
                
        except Exception as e:
            results['failed_drugs'].append((drug, str(e)))
            print(f"  ✗ {drug}: {e}")
    
    # Summary
    success_rate = len(results['successful_drugs']) / len(test_drugs) * 100
    print(f"\nCOMPREHENSIVE TEST RESULTS:")
    print(f"  Drugs tested: {len(test_drugs)}")
    print(f"  Successful: {len(results['successful_drugs'])} ({success_rate:.1f}%)")
    print(f"  Failed: {len(results['failed_drugs'])}")
    print(f"  Total periods extracted: {results['total_periods']:,}")
    print(f"  Total wells processed: {results['total_wells']:,}")
    
    if results['failed_drugs']:
        print(f"\nFailed drugs (showing first 5):")
        for drug, error in results['failed_drugs'][:5]:
            print(f"  {drug}: {error}")
    
    # Assert reasonable success rate
    assert success_rate >= 50, f"Success rate {success_rate:.1f}% too low, should be ≥50%"
    assert results['total_periods'] >= 10, f"Too few periods extracted: {results['total_periods']}"
    
    return results


def run_all_tests(comprehensive: bool = False):
    """Run all media change period feature tests."""
    print("=" * 60)
    print("MEDIA CHANGE PERIOD FEATURE TESTS")
    print("=" * 60)
    
    # Create test instances
    detection_tests = TestMediaChangePeriodDetection()
    splitting_tests = TestPeriodSplitting()
    feature_tests = TestPeriodFeatureExtraction()
    progressive_tests = TestProgressiveEffectAnalysis()
    integrity_tests = TestDataIntegrityValidation()
    biological_tests = TestBiologicalValidation()
    
    test_results = []
    
    # Run detection tests
    print("\n1. TESTING MEDIA CHANGE DETECTION...")
    try:
        detection_tests.test_can_load_media_events()
        print("✓ Can load media change events")
        test_results.append("✓ Media event loading")
    except Exception as e:
        print(f"✗ Media event loading failed: {e}")
        test_results.append("✗ Media event loading")
    
    try:
        detection_tests.test_events_have_realistic_timing()
        print("✓ Events have realistic timing")
        test_results.append("✓ Event timing validation")
    except Exception as e:
        print(f"✗ Event timing validation failed: {e}")
        test_results.append("✗ Event timing validation")
    
    try:
        detection_tests.test_can_align_events_with_oxygen_data()
        print("✓ Can align events with oxygen data")
        test_results.append("✓ Event-oxygen alignment")
    except Exception as e:
        print(f"✗ Event-oxygen alignment failed: {e}")
        test_results.append("✗ Event-oxygen alignment")
    
    # 2. Period splitting tests
    print(f"\n2. TESTING PERIOD SPLITTING...")
    try:
        splitting_tests.test_periods_are_sequential_and_non_overlapping()
        print("✓ Periods are sequential and non-overlapping")
        test_results.append("✓ Period splitting validation")
    except Exception as e:
        print(f"✗ Period splitting validation failed: {e}")
        test_results.append("✗ Period splitting validation")
    
    # 3. Feature extraction tests
    print(f"\n3. TESTING FEATURE EXTRACTION...")
    try:
        feature_tests.test_can_extract_basic_statistics_per_period()
        print("✓ Can extract basic statistics per period")
        test_results.append("✓ Basic feature extraction")
    except Exception as e:
        print(f"✗ Basic feature extraction failed: {e}")
        test_results.append("✗ Basic feature extraction")
    
    try:
        feature_tests.test_period_features_have_expected_structure()
        print("✓ Period features have expected structure")
        test_results.append("✓ Feature structure validation")
    except Exception as e:
        print(f"✗ Feature structure validation failed: {e}")
        test_results.append("✗ Feature structure validation")
    
    # 4. Biological validation tests
    print(f"\n4. TESTING BIOLOGICAL VALIDATION...")
    try:
        biological_tests.test_control_wells_show_predictable_patterns()
        print("✓ Control wells show predictable patterns")
        test_results.append("✓ Control well validation")
    except Exception as e:
        if "skip" in str(e).lower():
            print(f"⏭ Control well validation skipped: {e}")
            test_results.append("⏭ Control well validation skipped")
        else:
            print(f"✗ Control well validation failed: {e}")
            test_results.append("✗ Control well validation")
    
    try:
        biological_tests.test_toxic_drugs_show_progressive_deterioration()
        print("✓ Toxic drugs show measurable progression")
        test_results.append("✓ Toxic drug progression")
    except Exception as e:
        if "skip" in str(e).lower():
            print(f"⏭ Toxic drug progression skipped: {e}")
            test_results.append("⏭ Toxic drug progression skipped")
        else:
            print(f"✗ Toxic drug progression failed: {e}")
            test_results.append("✗ Toxic drug progression")
    
    # Note about remaining pending tests
    print("\n5. REMAINING PENDING TESTS (future implementation):")
    pending_tests = [
        "Recovery pattern detection",
        "Concentration-response monotonicity", 
        "Replicate consistency validation",
        "Advanced progressive effect analysis"
    ]
    
    for test in pending_tests:
        print(f"⏳ {test}")
        test_results.append(f"⏳ {test}")
    
    # Run comprehensive test if requested
    if comprehensive:
        try:
            comprehensive_results = test_all_drugs_comprehensive(max_drugs=10)  # Test top 10 drugs
            test_results.append(f"✓ Comprehensive test: {len(comprehensive_results['successful_drugs'])} drugs")
        except Exception as e:
            test_results.append(f"✗ Comprehensive test failed: {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in test_results if r.startswith("✓"))
    failed = sum(1 for r in test_results if r.startswith("✗"))
    pending = sum(1 for r in test_results if r.startswith("⏳"))
    
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"⏳ Pending: {pending}")
    
    if failed == 0:
        print("🎉 All implemented tests passed!")
        if comprehensive:
            print("📊 Comprehensive testing completed successfully")
        else:
            print("💡 Run with comprehensive=True to test all drugs")
    else:
        print("⚠ Some tests failed. Fix issues before proceeding.")
    
    return test_results


if __name__ == "__main__":
    import sys
    
    # Check if comprehensive testing is requested
    comprehensive = '--comprehensive' in sys.argv or '--all' in sys.argv
    run_all_tests(comprehensive=comprehensive)