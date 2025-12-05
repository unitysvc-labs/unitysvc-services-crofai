#!/usr/bin/env python3
"""
CROFAI Service Extractor
Pulls models from CROFAI API, fetches pricing, writes service.json and listing.json
"""

import os
import sys
import json
import requests
from decimal import Decimal
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class CrofAIModelExtractor:
    def __init__(self, api_key: str, api_base_url: str):
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Mozilla/5.0 (compatible; CrofAI-Service-Puller/1.0)",
            }
        )
        self.summary = {
            "total_models": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "skipped_models": 0,
            "extraction_date": datetime.utcnow().isoformat() + "Z",
        }

    def get_all_models(self, models_file: Optional[str] = None) -> List[Dict]:
        """Fetch all models from CROFAI API or load from file."""
        url = f"{self.api_base_url}/models"
        try:
            print(f"📡 Fetching from: {url}")
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])

            if not models:
                print(f"⚠️  No models found in response. Response keys: {data.keys()}")
                return []

            self.summary["total_models"] = len(models)
            models.sort(key=lambda x: x.get("id", ""))
            print(f"✅ Retrieved {len(models)} models from API")
            return models
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error fetching models: {e}")
            print(f"   URL attempted: {url}")
            print(f"   Tip: Save API response to a file and use --models-file option")
            return []
        except Exception as e:
            print(f"❌ Unexpected error fetching models: {e}")
            return []

    def create_pricing_info_structure(self, pricing_data: Dict) -> List[Dict]:
        """Create pricing structure from API pricing data."""
        return {
            "description": "Pricing Per 1M Tokens Input/Output",
            "input": str(Decimal(pricing_data["prompt"]) * 1000000),
            "output": str(Decimal(pricing_data["completion"]) * 1000000),
            "type": "one_million_tokens",
        }

    def create_service_data_structure(self, model_data: Dict) -> Dict:
        """Create service.json structure."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # Extract model details
        model_name = model_data.get("id", "").replace(":", "-")
        service_config = {
            "version": "",
            "schema": "service_v1",
            "time_created": timestamp,
            "name": model_name,
            "currency": "USD",
            "service_type": "llm",
            "display_name": model_data.get("name", ""),
            "description": "",
            "upstream_status": "ready",
            "details": {
                "context_length": model_data.get("context_length"),
                "max_completion_tokens": model_data.get("max_completion_tokens"),
                "quantization": model_data.get("quantization"),
                "speed": model_data.get("speed"),
                "created": model_data.get("created"),
            },
            "seller_price": {
                "type": "revenue_share",
                "percentage": "100.00",
                "description": "Pricing Per 1M Tokens",
            },
            "upstream_access_interface": {
                "name": "CROFAI API",
                "api_key": self.api_key,
                "base_url": self.api_base_url,
                "access_method": "http",
                "model_id": model_name,
            },
        }
        return service_config

    def create_listing_data_structure(self, pricing_data: Dict) -> Dict:
        """Create listing.json structure."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        listing_config = {
            "schema": "listing_v1",
            "seller_name": "svcreseller",
            "time_created": timestamp,
            "listing_status": "ready",
            "currency": "USD",
            "user_access_interfaces": [
                {
                    "name": "CROFAI API",
                    "base_url": "${GATEWAY_BASE_URL}/p/crofai",
                    "access_method": "http",
                    "documents": [
                        {
                            "category": "code_example",
                            "description": "Example code to use the model",
                            "file_path": "../../docs/code_example.py.j2",
                            "is_active": True,
                            "is_public": True,
                            "mime_type": "python",
                            "title": "Python code example",
                        },
                        {
                            "category": "code_example",
                            "description": "Example code to use the model",
                            "file_path": "../../docs/code_example_1.py.j2",
                            "is_active": True,
                            "is_public": True,
                            "mime_type": "python",
                            "title": "Python function calling code example",
                        },
                        {
                            "category": "code_example",
                            "description": "Example code to use the model",
                            "file_path": "../../docs/code_example.js.j2",
                            "is_active": True,
                            "is_public": True,
                            "mime_type": "javascript",
                            "title": "JavaScript code example",
                        },
                        {
                            "category": "code_example",
                            "description": "Example code to use the model",
                            "file_path": "../../docs/code_example.sh.j2",
                            "is_active": True,
                            "is_public": True,
                            "mime_type": "bash",
                            "title": "cURL code example",
                        },
                        {
                            "category": "getting_started",
                            "description": "",
                            "file_path": "../../docs/description.md",
                            "is_active": True,
                            "is_public": True,
                            "mime_type": "markdown",
                            "title": "How to use this model",
                        },
                    ],
                    "name": "Provider API",
                }
            ],
            "customer_price": self.create_pricing_info_structure(pricing_data),
        }
        return listing_config

    def write_service_files(
        self, service_data: Dict, output_dir: Path, raw_pricing: Dict
    ):
        """Write service.json and listing.json files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write service.json
        service_file = output_dir / "service.json"
        with open(service_file, "w", encoding="utf-8") as f:
            json.dump(service_data, f, indent=2)

        # Write listing.json with raw pricing data
        listing_data = self.create_listing_data_structure(raw_pricing)
        listing_file = output_dir / "listing.json"
        with open(listing_file, "w", encoding="utf-8") as f:
            json.dump(listing_data, f, indent=2)

        print(f"  ✅ Written files to {output_dir}")

    def process_all_models(
        self,
        output_dir: str = "services",
        specific_models: Optional[List[str]] = None,
        force: bool = False,
        dry_run: bool = False,
        models_file: Optional[str] = None,
    ):
        """Process all models and create service folders."""
        base_path = Path(output_dir)

        if specific_models:
            # Fetch full data for specific models
            all_models = self.get_all_models(models_file)
            models = [m for m in all_models if m.get("id") in specific_models]
        else:
            models = self.get_all_models(models_file)

        for i, model_data in enumerate(models, 1):
            model_name = model_data.get("id").replace(":", "-")
            if not model_name:
                print(f"  ⚠️  Skipping model without ID")
                continue

            print(f"\n[{i}/{len(models)}] Processing: {model_name}")

            # Create directory name from model ID (sanitize for filesystem)
            model_dir = base_path / model_name

            # Check if already exists
            if not force and (model_dir / "service.json").exists():
                print(
                    f"  ⏭️  Skipping {model_name} (already exists, use --force to overwrite)"
                )
                self.summary["skipped_models"] += 1
                continue

            # Create service structure
            try:
                service_config = self.create_service_data_structure(model_data)
                raw_pricing = model_data.get("pricing", {})

                if dry_run:
                    print(f"  📝 [DRY-RUN] Would create directory: {model_dir}")
                    print(f"  📝 [DRY-RUN] Would write service.json and listing.json")
                    if raw_pricing:
                        print(
                            f"  📝 [DRY-RUN] Pricing: prompt={raw_pricing.get('prompt')}, completion={raw_pricing.get('completion')}"
                        )
                else:
                    self.write_service_files(service_config, model_dir, raw_pricing)
                    self.summary["successful_extractions"] += 1
                    if raw_pricing:
                        print(
                            f"  💰 Pricing included: prompt=${raw_pricing.get('prompt')}, completion=${raw_pricing.get('completion')}"
                        )
            except Exception as e:
                print(f"  ❌ Error processing {model_id}: {e}")
                import traceback

                traceback.print_exc()
                self.summary["failed_extractions"] += 1

        # Print summary
        print("\n" + "=" * 60)
        print("📊 EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Total models: {self.summary['total_models']}")
        print(f"Successfully processed: {self.summary['successful_extractions']}")
        print(f"Failed: {self.summary['failed_extractions']}")
        print(f"Skipped: {self.summary['skipped_models']}")
        if dry_run:
            print("\n⚠️  DRY-RUN mode - no files were written")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CROFAI services and pricing")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="services",
        help="Output directory for service files",
    )
    parser.add_argument("--models", nargs="+", help="Specific model IDs to process")
    parser.add_argument(
        "--models-file",
        type=str,
        help="JSON file containing models data (skips API call)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing service files"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without writing files"
    )
    args = parser.parse_args()

    # Get credentials from environment
    api_key = os.environ.get("CROFAI_API_KEY", "")
    api_base_url = os.environ.get("CROFAI_API_BASE_URL", "https://ai.nahcrof.com/v2")

    # API key not required if using models file
    if not args.models_file and not api_key:
        print("❌ Error: CROFAI_API_KEY environment variable not set")
        print("\nUsage:")
        print("  export CROFAI_API_KEY='your-api-key'")
        print("  python extract_crofai_services.py")
        print("\nOr use a models file:")
        print("  python extract_crofai_services.py --models-file models.json")
        sys.exit(1)

    print(f"🚀 Starting CROFAI Service Extractor")
    print(f"📁 Output directory: {args.output_dir}")
    if args.models_file:
        print(f"📂 Models file: {args.models_file}")
    else:
        print(f"🔑 API Base URL: {api_base_url}")
    print()

    extractor = CrofAIModelExtractor(api_key, api_base_url)
    extractor.process_all_models(
        args.output_dir,
        specific_models=args.models,
        force=args.force,
        dry_run=args.dry_run,
        models_file=args.models_file,
    )
