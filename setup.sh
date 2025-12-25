#!/bin/bash
# Thames Connectivity Analysis - Project Setup Script
set -e  # Exit on any error

echo "Project Setup"
echo "================================================"

# Get script directory (project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "📁 Project root: $PROJECT_ROOT"

# Create project directory structure
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed/{transit,boundaries,spatial,metrics}}
mkdir -p artifacts figures tables docs/qa tests notebooks

echo "✅ Directory structure created"


# Check for UV package manager
if ! command -v uv &> /dev/null; then
    echo "❌ UV package manager not found"
    echo "   Install UV: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "✅ UV package manager found"

# Install dependencies
echo "📦 Installing project dependencies..."
if [ -f "pyproject.toml" ]; then
    uv sync
    echo "✅ Dependencies installed via UV"
else
    echo "❌ pyproject.toml not found"
    exit 1
fi

# Display environment information
echo ""
echo "Environment Information"
echo "========================="
echo "Python version: $(uv run python --version)"
echo "Platform: $(uname -s -r -m)"

echo ""
echo "📦 Key Dependencies"
echo "=================="
for package in pandas numpy networkx geopandas shapely pyproj scipy statsmodels matplotlib; do
    version=$(uv run python -c "import $package; print($package.__version__)" 2>/dev/null || echo "not installed")
    printf "%-15s %s\n" "$package:" "$version"
done

echo ""
echo "🎉 Setup Complete!"