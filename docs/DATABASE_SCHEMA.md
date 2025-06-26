# Database Schema Documentation

This document provides the complete schema information for the main tables used in the organoid DILI prediction project.

## Custom Types (Enums)

### pipeline_status
- `complete`
- `submitted`
- `failed`
- `running`
- `null`
- `created`
- `deleted`
- `cancelled`
- `partially_complete`

### status
- `created`
- `image_qc`
- `metabolic_qc`
- `raw_data`
- `ended`

### state
- `active`
- `abandoned`
- `completed`

## Table Schemas

### 1. processed_data
Primary time series data containing oxygen measurements for each well over time.

| Column | Data Type | PostgreSQL Type | Nullable | Default | Description |
|--------|-----------|-----------------|----------|---------|-------------|
| id | bigint | int8 | NO | - | Primary key |
| plate_id | uuid | uuid | NO | - | Foreign key to plate_table |
| well_number | smallint | int2 | NO | - | Well number on plate |
| timestamp | timestamp with time zone | timestamptz | NO | - | Measurement timestamp |
| median_o2 | real | float4 | YES | - | Median oxygen percentage |
| cycle_time_stamp | timestamp with time zone | timestamptz | YES | - | Cycle timestamp |
| cycle_num | smallint | int2 | YES | - | Cycle number |
| is_excluded | boolean | bool | YES | false | Exclusion flag |
| exclusion_reason | text | text | YES | - | Reason for exclusion |
| excluded_by | uuid | uuid | YES | - | User who excluded |
| excluded_at | timestamp with time zone | timestamptz | YES | - | Exclusion timestamp |

### 2. drugs
Drug information including DILI risk classifications and pharmacokinetic properties.

| Column | Data Type | PostgreSQL Type | Nullable | Default | Description |
|--------|-----------|-----------------|----------|---------|-------------|
| id | bigint | int8 | NO | - | Primary key |
| created_at | timestamp with time zone | timestamptz | NO | now() | Creation timestamp |
| drug | text | text | NO | - | Drug name (unique) |
| c_max | text | text | YES | - | Peak plasma concentration |
| dili | text | text | YES | - | DILI classification |
| binary_dili | boolean | bool | YES | - | Binary DILI flag |
| likelihood | text | text | YES | - | DILI likelihood |
| logp | real | float4 | YES | - | LogP value |
| misc | jsonb | jsonb | YES | - | Miscellaneous data |
| smiles | text | text | YES | - | SMILES notation |
| severity | smallint | int2 | YES | - | DILI severity score |
| molecular_weight | real | float4 | YES | - | Molecular weight |
| atc | text | text | YES | - | ATC classification |
| experimental_names | text[] | _text | YES | - | Alternative names array |
| cmax_oral_m | real | float4 | YES | - | Oral Cmax (molar) |
| cmax_iv_m | real | float4 | YES | - | IV Cmax (molar) |
| cmax_topical_m | real | float4 | YES | - | Topical Cmax (molar) |
| cmax_dose | text | text | YES | - | Cmax dose info |
| cmax_notes | text | text | YES | - | Cmax notes |
| active_metabolites | text | text | YES | - | Active metabolites |
| metabolite_contribution | text | text | YES | - | Metabolite contribution |
| cmax_metabolite_m | real | float4 | YES | - | Metabolite Cmax (molar) |
| cmin_oral_m | real | float4 | YES | - | Oral Cmin (molar) |
| cmin_therapeutic_threshold | real | float4 | YES | - | Therapeutic threshold |
| cmin_notes | text | text | YES | - | Cmin notes |
| food_effect_cmax_ratio | real | float4 | YES | - | Food effect on Cmax |
| food_effect_auc_ratio | real | float4 | YES | - | Food effect on AUC |
| food_requirements | text | text | YES | - | Food requirements |
| cns_penetration_ratio | real | float4 | YES | - | CNS penetration |
| blood_brain_barrier | boolean | bool | YES | - | BBB penetration flag |
| cns_notes | text | text | YES | - | CNS notes |
| therapeutic_min_m | real | float4 | YES | - | Min therapeutic (molar) |
| therapeutic_max_m | real | float4 | YES | - | Max therapeutic (molar) |
| toxic_threshold_m | real | float4 | YES | - | Toxic threshold (molar) |
| hepatic_impairment_factor | real | float4 | YES | - | Hepatic impairment factor |
| renal_impairment_factor | real | float4 | YES | - | Renal impairment factor |
| special_population_notes | text | text | YES | - | Special population notes |
| pk_references | text | text | YES | - | PK data references |
| therapeutic_range_min_m | real | float4 | YES | - | Min therapeutic range (molar) |
| therapeutic_range_max_m | real | float4 | YES | - | Max therapeutic range (molar) |
| therapeutic_range_notes | text | text | YES | - | Therapeutic range notes |
| half_life_hours | real | float4 | YES | - | Half-life (hours) |
| volume_distribution_l_kg | real | float4 | YES | - | Volume of distribution |
| protein_binding_percent | real | float4 | YES | - | Protein binding % |
| bioavailability_percent | real | float4 | YES | - | Bioavailability % |
| clearance_l_hr_kg | real | float4 | YES | - | Clearance rate |
| metabolism_cyp_enzymes | text | text | YES | - | CYP enzymes |
| dili_risk_category | text | text | YES | - | DILI risk category |
| dili_detailed_notes | text | text | YES | - | Detailed DILI notes |
| monitoring_recommendations | text | text | YES | - | Monitoring recommendations |
| specific_toxicity_flags | text | text | YES | - | Toxicity flags |
| differentiation_syndrome_risk | boolean | bool | YES | false | Differentiation syndrome flag |
| hepatotoxicity_boxed_warning | boolean | bool | YES | false | Boxed warning flag |
| therapeutic_drug_monitoring | boolean | bool | YES | false | TDM flag |
| tmax_hours | real | float4 | YES | - | Time to max concentration |
| auc_0_24h_m_h | real | float4 | YES | - | AUC 0-24h (molar*h) |
| steady_state_days | real | float4 | YES | - | Days to steady state |
| accumulation_ratio | real | float4 | YES | - | Accumulation ratio |
| lag_time_hours | real | float4 | YES | - | Lag time (hours) |
| formulation_notes | text | text | YES | - | Formulation notes |
| pediatric_dose_mg_m2 | real | float4 | YES | - | Pediatric dose |
| geriatric_adjustment | real | float4 | YES | - | Geriatric adjustment |
| adult_dose_mg | real | float4 | YES | - | Adult dose (mg) |
| adult_dose_frequency | text | text | YES | - | Adult dose frequency |
| dose_notes | text | text | YES | - | Dose notes |
| pediatric_dose_frequency | text | text | YES | - | Pediatric dose frequency |
| cmax_dose_mg | real | float4 | YES | - | Cmax dose (mg) |
| cmax_condition | text | text | YES | - | Cmax condition |
| auc_0_24h_ng_h_ml | real | float4 | YES | - | AUC 0-24h (ng*h/mL) |
| auc_notes | text | text | YES | - | AUC notes |
| timing_notes | text | text | YES | - | Timing notes |

### 3. event_table
Events that occur during experiments (drug additions, media changes, etc).

| Column | Data Type | PostgreSQL Type | Nullable | Default | Description |
|--------|-----------|-----------------|----------|---------|-------------|
| id | uuid | uuid | NO | gen_random_uuid() | Primary key |
| plate_id | uuid | uuid | NO | - | Foreign key to plate_table |
| created_at | timestamp with time zone | timestamptz | NO | now() | Creation timestamp |
| occurred_at | timestamp with time zone | timestamptz | NO | now() | Event occurrence time |
| uploaded_by | uuid | uuid | NO | - | User who uploaded |
| title | text | text | YES | - | Event title |
| description | text | text | YES | - | Event description |
| is_excluded | boolean | bool | NO | false | Exclusion flag |
| exclusion_reason | text | text | YES | - | Reason for exclusion |

### 4. well_map_data
Well metadata including drug, concentration, and tissue information.

| Column | Data Type | PostgreSQL Type | Nullable | Default | Description |
|--------|-----------|-----------------|----------|---------|-------------|
| id | bigint | int8 | NO | - | Primary key |
| plate_id | uuid | uuid | NO | - | Foreign key to plate_table |
| well_number | smallint | int2 | NO | - | Well number on plate |
| drug | text | text | NO | - | Drug name |
| concentration | real | float4 | NO | - | Drug concentration |
| units | text | text | NO | - | Concentration units |
| tissue | text | text | NO | - | Tissue type |
| sample | text | text | NO | - | Sample identifier |
| description | text | text | YES | - | Well description |
| is_excluded | boolean | bool | NO | false | Exclusion flag |
| exclusion_reason | text | text | YES | - | Reason for exclusion |
| exclusion_level | smallint | int2 | YES | - | Exclusion severity level |

### 5. plate_table
Plate metadata and configuration.

| Column | Data Type | PostgreSQL Type | Nullable | Default | Description |
|--------|-----------|-----------------|----------|---------|-------------|
| id | uuid | uuid | NO | gen_random_uuid() | Primary key |
| name | text | text | NO | - | Plate name (unique) |
| created_at | timestamp without time zone | timestamp | NO | now() | Creation timestamp |
| updated_at | timestamp without time zone | timestamp | YES | now() | Update timestamp |
| created_by | uuid | uuid | YES | - | User who created |
| deleted | boolean | bool | YES | false | Deletion flag |
| status | status (enum) | status | NO | 'created' | Plate status |
| state | state (enum) | state | NO | 'active' | Plate state |
| tissue | text | text | YES | 'Liver' | Tissue type |
| description | text | text | YES | - | Plate description |
| plate_size | smallint[] | _int2 | NO | '{16,24}' | Plate dimensions array |
| qc_values | jsonb | jsonb | YES | - | QC values JSON |
| qc_thresholds | jsonb | jsonb | NO | (see default below) | QC thresholds JSON |
| internal_notes | jsonb | jsonb | NO | '{}' | Internal notes JSON |

Default qc_thresholds:
```json
{
  "image_qc": {
    "largest_size": {"max": 5, "min": 0.5},
    "num_of_organoids": {"max": 5, "min": 1}
  },
  "metabolic_qc": {
    "o2": {"max": 150, "min": -50},
    "snr": {"max": null, "min": 0.7}
  }
}
```

## Data Type Mappings for DuckDB

When creating tables in DuckDB, use these mappings:

| PostgreSQL Type | DuckDB Type | Notes |
|-----------------|-------------|-------|
| bigint/int8 | BIGINT | 64-bit integer |
| smallint/int2 | SMALLINT | 16-bit integer |
| real/float4 | REAL | 32-bit float |
| text | VARCHAR | Variable-length string |
| uuid | UUID | UUID type |
| timestamp with time zone | TIMESTAMPTZ | Timestamp with timezone |
| timestamp without time zone | TIMESTAMP | Timestamp without timezone |
| boolean/bool | BOOLEAN | True/false |
| jsonb | JSON | JSON data |
| ARRAY | LIST | Array types |
| USER-DEFINED (enum) | VARCHAR | Store as string |

## Important Notes

1. **UUID Handling**: PostgreSQL UUIDs should be stored as UUID type in DuckDB (supported in recent versions) or as VARCHAR if UUID type is not available.

2. **Array Types**: 
   - `experimental_names` in drugs table is `text[]` (array of text)
   - `plate_size` in plate_table is `smallint[]` (array of smallint)
   - These should be stored as LIST types in DuckDB

3. **JSONB Fields**: The `misc` field in drugs table and `qc_values`, `qc_thresholds`, `internal_notes` in plate_table are JSONB and should be stored as JSON in DuckDB.

4. **Enum Types**: The `status` and `state` columns in plate_table use custom enum types. These should be stored as VARCHAR in DuckDB with appropriate CHECK constraints if needed.

5. **Timezone Handling**: Most timestamps use `timestamptz` (with timezone) except `created_at` and `updated_at` in plate_table which use `timestamp` (without timezone).