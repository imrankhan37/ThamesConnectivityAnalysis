# Thames Connectivity Analysis

A network analysis of London's transit system examining connectivity patterns, bottlenecks, resilience, and equity across the Thames divide.

## Quick Start

### Analysis Only 

**All preprocessed data is provided.** You can skip directly to the analysis notebooks:

```bash
# 1. Clone repository
git clone https://github.com/imrankhan37/ThamesConnectivityAnalysis.git
cd ThamesConnectivityAnalysis

# 2. Setup environment (installs dependencies only)
chmod +x setup.sh
./setup.sh

# 3. Unzip provided raw data (if needed)
bash scripts/unzip_raw_data.sh

# 4. Run analysis notebooks directly
uv run jupyter notebook notebooks/
```

**Time required: ~5 minutes total** (1-2 minutes per notebook)

### Prerequisites
- [UV package manager](https://docs.astral.sh/uv/getting-started/installation/) (handles Python 3.11 automatically)


### ✅ Pre-Provided Data (No Processing Required)

The repository includes **all processed outputs** from Phases 1-3:

| Phase | Output Location | Description | Files Included |
|-------|----------------|-------------|----------------|
| **Raw Data** | `data/raw/` | All source datasets | ✓ TfL stations/routes<br>✓ Thames centerline<br>✓ London boundaries<br>✓ OS Open Rivers |
| **Phase 2 Outputs** | `data/processed/transit/` | Network construction | ✓ stations.csv<br>✓ edges.csv<br>✓ stations_london.csv<br>✓ edges_london.csv |
| **Phase 3 Outputs** | `data/processed/spatial/` | Spatial processing | ✓ station_bank.csv<br>✓ edge_is_thames_crossing.csv<br>✓ crossing_count.csv |

Only the **analysis notebooks** (Phase 4):
1. `01_network_metrics.ipynb` - Basic network properties
2. `02_h1_bottleneck_analysis.ipynb` - CRREB analysis
3. `03_h2_resilience_analysis.ipynb` - Resilience testing
4. `04_h1_validation.ipynb` - Statistical validation

## Full Pipeline

### Complete Pipeline Execution (Optional)

If you wish to reproduce the entire pipeline from scratch:

```bash
# Phase 1: Data Ingestion (~2-3 minutes)
uv run python scripts/phases/ingest_data.py

# Phase 2: Network Construction (~1-2 minutes)
uv run python scripts/phases/transform_load_network_data.py

# Phase 3: Spatial Processing (~1-2 minutes)
uv run python scripts/phases/spatial_processor.py

# Phase 4: Analysis Notebooks (~1-2 minutes each)
uv run jupyter notebook notebooks/
```

### Phase 1 Detail: Data Ingestion

**Main script:** `scripts/phases/ingest_data.py`

**Subscripts called:**
- `scripts/data_ingestion/01_ingest_transit.py` - TfL API data
- `scripts/data_ingestion/02_ingest_thames.py` - Thames geometry
- `scripts/data_ingestion/03_ingest_boundary.py` - London boundaries

**Note:** This phase requires internet access for API calls. All outputs are already provided in `data/raw/`.

## Project Structure

### Core Phases
1. **Data Ingestion** (`scripts/phases/ingest_data.py`)
   - TfL transit data (stations, routes, sequences)
   - Thames centerline geometry
   - London boundary polygon (for London-only filtering/plots)

2. **Network Construction** (`scripts/phases/transform_load_network_data.py`)
   - Station-edge graph from TfL route sequences
   - London boundary filtering
   - Network validation and visualization

3. **Spatial Processing** (`scripts/phases/spatial_processor.py`)
   - Thames bank classification
   - Cross-river connectivity detection

4. **Analysis Notebooks**
   - `01_network_metrics.ipynb`: Basic network properties and centrality analysis
   - `02_h1_bottleneck_analysis.ipynb`: Cross-River Restricted Edge Betweenness (CRREB) analysis
   - `03_h2_resilience_analysis.ipynb`: Network resilience to targeted crossing disruptions
   - `04_h1_validation.ipynb`: Statistical validation using matched random-set null model

### Directory Structure
```
ThamesConnectivityAnalysis/
├── data/
│   ├── raw/                # Original raw datasets
│   ├── processed/          # Cleaned, validated datasets
│   │   ├── transit/       
│   │   ├── boundaries/    
│   │   ├── spatial/       
│   │   └── _meta/         
│   └── analysis/          
├── src/                    # Python source code modules
│   ├── core/               # Config, logging, base utilities
│   ├── data_processing/    # Data ingestion and cleaning logic
│   ├── geo/                # Spatial operations and helpers
│   ├── connectivity/       # Network analysis algorithms
│   ├── graph/              
│   └── vis/                
├── scripts/
│   └── phases/             
├── notebooks/             
├── artifacts/              
└── config/                 # Pipeline and environment configuration files
```


### Key Outputs
- **Network Data**: `data/processed/transit/` (stations, edges, London subset)
- **Spatial Mappings**: `data/processed/spatial/` (LSOA joins, bank classifications)
- **Visualistions**: `artifacts/` (network maps, analysis figures)

## Data Sources

### Required raw data (OS Open Rivers is manual)
- **TfL API (auto-ingested)**: stop points, route sequences, line information (downloaded by `scripts/phases/ingest_data.py`).
- **ONS Geography (auto-ingested)**: London boundary polygon (downloaded by `scripts/phases/ingest_data.py`).
- **OS Open Rivers (manual)**: you must download the OS Open Rivers GeoPackage and place it at:
  - `data/raw/oprvrs_gb.gpkg`

Because of upload-size constraints, we also provide the raw GeoPackage in compressed form:
- `data/raw/oprvrs_gb.gpkg.zip`

To use the zipped version, run:

```bash
bash scripts/unzip_raw_data.sh
```

### Data Pipeline
All raw datasets are preprocessed through validated schemas ensuring:
- Consistent coordinate reference systems (WGS84 → BNG)
- Type safety with pandas nullable dtypes
- Spatial topology validation
- Comprehensive QA reporting

## Hypotheses Tested

### H1: Bottleneck Identification
Critical stations whose removal disproportionately impacts network connectivity, particularly at Thames crossing points.

### H2: Resilience Analysis
Network vulnerability to cascading failures, with focus on cross-Thames connectivity preservation.

## Reproducibility Notes

### Hardware Specifications

**Development/Testing Environment:**
- **CPU**: 12th Gen Intel Core i7-1260P (16 cores)
- **RAM**: 16GB
- **Platform**: WSL2 Ubuntu 22.04 on Windows 11
- **Python**: 3.11.10 (managed by UV)

**Minimum Requirements:**
- **CPU**: Any modern x64 processor (4+ cores recommended)
- **RAM**: 4GB minimum, 8GB+ recommended for spatial operations
- **Disk**: ~500MB for all data and outputs

### Dependencies
- All dependencies pinned in `pyproject.toml` with lockfile `uv.lock`
- Python 3.11 enforced via UV package manager
- Geographic operations use EPSG:27700 (British National Grid)
- Random seed: 20250101 (set in notebooks for reproducibility)