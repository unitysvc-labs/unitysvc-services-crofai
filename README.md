# UnitySVC Services Template

**A template repository for digital service providers to publish their services on the UnitySVC platform.**

Use this template to create your own service provider repository with automated validation and publishing workflows.

📚 **[Full Documentation](https://unitysvc-services.readthedocs.io)** | 🚀 **[Getting Started Guide](https://unitysvc-services.readthedocs.io/en/latest/getting-started/)** | 📖 **[CLI Reference](https://unitysvc-services.readthedocs.io/en/latest/cli-reference/)**

## 🚀 Quick Start

### 1. Create Your Repository from Template

1. Click the **"Use this template"** button at the top of this repository
2. Choose a name for your repository (e.g., `unitysvc-services-yourcompany`)
3. Clone your new repository locally

### 2. Install unitysvc-services CLI

```bash
pip install unitysvc-services
```

See the [CLI Reference](https://unitysvc-services.readthedocs.io/en/latest/cli-reference/) for all available commands.

### 3. Customize Your Data

Replace the example data in the `data/` directory with your actual service information:

**Required changes:**

- [ ] Update `data/seller.json` with your seller/marketplace information
- [ ] Update `data/${provider}/provider.toml` with your company information
- [ ] Update `data/${provider}/README.md` with your company description
- [ ] Update `data/${provider}/docs/` with your actual documentation and code examples
- [ ] Replace example services in `data/${provider}/services/` with your actual services

**Recommended naming:**

- Provider name: Use your company name (e.g., "acme-corp")
- Service names: Use descriptive names (e.g., "gpt-4-turbo", "claude-3-opus")
- Keep names lowercase with hyphens (e.g., "my-service-name")

See [Data Structure Documentation](https://unitysvc-services.readthedocs.io/en/latest/data-structure/) for complete details on file organization and naming rules.

### 4. Validate Your Data

Before committing changes:

```bash
# Validate all files
unitysvc_services validate

# Format files to match requirements
unitysvc_services format
```

## 📁 Repository Structure

```
data/
├── seller.json                      # Seller metadata (ONE per repo)
└── ${provider_name}/                # Directory name must match provider name
    ├── provider.toml                # Provider metadata
    ├── README.md                    # Provider documentation
    ├── docs/                        # Code examples and descriptions
    │   ├── code-example.py
    │   ├── code-example.js
    │   ├── code-example.sh
    │   └── description.md
    └── services/                    # Service definitions
        └── ${service_name}/         # Directory name must match service name
            ├── service.json         # Service offering (technical specs)
            └── listing-${seller}.json  # Service listing (user-facing info)
```

**Important**:
- Directory names must match the `name` fields in their respective files
- Only ONE seller file per repository
- Both service offerings and listings go under the same `services/${service_name}/` directory

See [Data Structure Documentation](https://unitysvc-services.readthedocs.io/en/latest/data-structure/) for complete details.

## 🤖 GitHub Actions Workflows

This template includes three automated workflows to streamline your development process:

### 1. Validate Data Workflow

**File**: `.github/workflows/validate-data.yml`
**Triggers**: Every pull request and push to main

Automatically validates all data files to ensure:
- Schema compliance
- File references exist
- Directory name consistency
- Service name uniqueness
- Valid seller references

This prevents invalid data from being merged.

### 2. Format Check Workflow

**File**: `.github/workflows/format-check.yml`
**Triggers**: Every pull request and push to main

Checks that all JSON and TOML files are properly formatted:
- JSON files have 2-space indentation
- Keys are sorted alphabetically
- Files end with newlines
- No trailing whitespace

Run `unitysvc_services format` locally to auto-fix formatting issues.

### 3. Publish Data Workflow

**File**: `.github/workflows/publish-data.yml`
**Triggers**: Push to main branch (after PR merge)

Automatically publishes your data to the UnitySVC backend in the correct order:

1. Providers
2. Sellers
3. Service Offerings
4. Service Listings

**Setup Required**: Configure GitHub secrets (see below).

### 4. Populate Services Workflow (Optional)

**File**: `.github/workflows/populate-services.yml`
**Triggers**:
- Daily at 2 AM UTC (scheduled)
- Manual trigger via GitHub Actions UI

Automatically updates service data by:

1. Running `unitysvc_services populate` to execute provider-specific update scripts
2. Formatting generated files
3. Creating a pull request with changes (if any)

**How It Works**:

The workflow scans your `data/` directory for provider files with a `services_populator` section:

```toml
# data/my-provider/provider.toml
[services_populator]
command = "populate_services.py"

[provider_access_info]
API_KEY = "your-provider-api-key"
API_ENDPOINT = "https://api.provider.com/v1"
```

The `populate` command will:
- Find all providers with `services_populator` configured
- Execute the specified command (e.g., `populate_services.py`)
- Pass environment variables from `provider_access_info`
- Generate/update service files automatically

**To Use This Workflow**:

1. Add a `services_populator` section to your provider.toml
2. Create a populate script that generates service data files
3. Configure provider access credentials in `provider_access_info`
4. The workflow will run daily or can be triggered manually

See [Automated Workflow Documentation](https://unitysvc-services.readthedocs.io/en/latest/workflows/#automated-workflow) for details on writing populate scripts.

**Example Populate Script**:

```python
#!/usr/bin/env python3
"""Populate services from provider API."""
import os
import json
from pathlib import Path

# Get credentials from environment (injected from provider_access_info)
api_key = os.environ.get("API_KEY")
api_endpoint = os.environ.get("API_ENDPOINT")

# Fetch services from provider API
services = fetch_services(api_endpoint, api_key)

# Generate service files
for service in services:
    service_dir = Path(f"services/{service['name']}")
    service_dir.mkdir(parents=True, exist_ok=True)

    # Write service offering
    with open(service_dir / "service.json", "w") as f:
        json.dump(generate_service_data(service), f, indent=2)
```

## 🔐 GitHub Secrets Configuration

To enable automatic publishing when changes are merged to `main`, configure the following GitHub secrets:

### Step 1: Navigate to Repository Settings

1. Go to your repository on GitHub
2. Click **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**

### Step 2: Add Secrets

Click **New repository secret** and add the following:

#### `UNITYSVC_BACKEND_URL`

- **Description**: The UnitySVC backend API URL
- **Example values**:
  - Production: `https://api.unitysvc.com/api/v1`
  - Staging: `https://staging.unitysvc.com/api/v1`
  - Development: `https://main.devel.unitysvc.com/api/v1`

#### `UNITYSVC_API_KEY`

- **Description**: API key for authenticating with the UnitySVC backend
- **How to obtain**:
  1. Log in to the UnitySVC platform (username should match the "seller" in service listings)
  2. Navigate to **Settings** → **API Keys**
  3. Generate a new API key for your organization
  4. Copy the key (you won't be able to see it again)

### Step 3: Verify Configuration

After adding the secrets:

1. Make a change to a file in the `data/` directory
2. Create a pull request
3. Merge the pull request to `main`
4. Check the **Actions** tab to see the publishing workflow run
5. Verify that data appears on your UnitySVC backend

## 📝 Development Workflow

### Basic Commands

```bash
# Create new data files
unitysvc_services init provider my-provider
unitysvc_services init seller my-marketplace
unitysvc_services init offering my-service
unitysvc_services init listing my-listing

# Validate data locally
unitysvc_services validate

# Format data files
unitysvc_services format

# List local data
unitysvc_services list providers
unitysvc_services list sellers
unitysvc_services list offerings
unitysvc_services list listings

# Publish manually (optional)
export UNITYSVC_BACKEND_URL="https://api.unitysvc.com/api/v1"
export UNITYSVC_API_KEY="your-api-key"
unitysvc_services publish providers
unitysvc_services publish sellers
unitysvc_services publish offerings
unitysvc_services publish listings
```

For complete CLI documentation, see [CLI Reference](https://unitysvc-services.readthedocs.io/en/latest/cli-reference/).

### Pre-commit Hooks (Recommended)

Install pre-commit hooks to automatically validate and format files before each commit:

```bash
pip install pre-commit
pre-commit install
```

The hooks will automatically format JSON files and validate data on every commit.

## 🔄 Contributing

1. Create a new branch for your changes
2. Make your changes to files in the `data/` directory
3. Run `unitysvc_services validate` to check your changes
4. Run `unitysvc_services format` to format files
5. Commit your changes (pre-commit hooks will run automatically)
6. Push your branch and create a pull request
7. Once approved and merged, data will be automatically published

## 📚 Documentation

For detailed information:

- **[Getting Started](https://unitysvc-services.readthedocs.io/en/latest/getting-started/)** - Installation and first steps
- **[Data Structure](https://unitysvc-services.readthedocs.io/en/latest/data-structure/)** - File organization rules
- **[Workflows](https://unitysvc-services.readthedocs.io/en/latest/workflows/)** - Manual and automated patterns
- **[CLI Reference](https://unitysvc-services.readthedocs.io/en/latest/cli-reference/)** - All commands and options
- **[File Schemas](https://unitysvc-services.readthedocs.io/en/latest/file-schemas/)** - Schema specifications

## 💡 Tips

- **Keep service names consistent**: Use the same name across `service.json`, `listing-*.json`, and directory names
- **Test locally first**: Always run `unitysvc_services validate` before pushing
- **Use pre-commit hooks**: They catch formatting and validation errors early
- **Document your services**: Good documentation helps users understand and adopt your services
- **Publishing order matters**: Always publish in order: providers → sellers → offerings → listings

## 📞 Support

For issues or questions:

- **UnitySVC Services SDK**: https://github.com/unitysvc/unitysvc-services
- **Documentation**: https://unitysvc-services.readthedocs.io
- **Issues**: Open an issue in this repository

## 📜 License

This template is provided under the MIT License. Service data you add is subject to your own licensing terms.
