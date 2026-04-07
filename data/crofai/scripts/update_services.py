#!/usr/bin/env python3
"""
update_services.py - Extract model data from CrofAI API and generate service files

Usage:
  python update_services.py [output_dir]
  python update_services.py --models model1 model2
  python update_services.py --force
"""

import os
import sys
import json
import requests
import argparse
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from jinja2 import Environment, FileSystemLoader, StrictUndefined


PROVIDER_NAME = "crofai"
PROVIDER_DISPLAY_NAME = "CrofAI"
ENV_API_KEY_NAME = "CROFAI_API_KEY"


def _sanitize_header_value(value: str) -> str:
    """Strip smart/curly quotes and non-latin-1 chars that break HTTP headers."""
    for bad, good in [("\u201c", '"'), ("\u201d", '"'), ("\u2018", "'"), ("\u2019", "'")]:
        value = value.replace(bad, good)
    value = value.encode("latin-1", errors="ignore").decode("latin-1").strip()
    value = value.strip('"').strip("'")
    return value


def derive_service_type(model_id: str) -> str:
    mid = model_id.lower()
    if any(k in mid for k in ["embed", "embedding"]):
        return "embedding"
    if any(k in mid for k in ["flux", "stable-diffusion", "sdxl"]):
        return "image_generation"
    return "llm"


class CrofAIModelExtractor:
    def __init__(self, api_key: str, api_base_url: str, templates_dir: Path):
        api_key = _sanitize_header_value(api_key)
        self.api_key = api_key
        self.api_base_url = (api_base_url or "https://ai.nahcrof.com/v2").strip()
        self.templates_dir = templates_dir
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
            "extraction_date": datetime.now().isoformat(),
            "force_mode": False,
            "processing_limit": None,
        }

        self.jinja_env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        self.jinja_env.filters["tojson"] = lambda v: json.dumps(v)

    # ------------------------------------------------------------------
    # Model listing
    # ------------------------------------------------------------------

    def get_all_models(self) -> List[Dict]:
        """Fetch all models from CrofAI API."""
        url = f"{self.api_base_url}/models"
        try:
            print(f"📡 Fetching from: {url}")
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            if not models:
                print(f"⚠️  No models found. Keys: {list(data.keys())}")
                return []
            self.summary["total_models"] = len(models)
            models.sort(key=lambda x: x.get("id", ""))
            print(f"✅ Retrieved {len(models)} models")
            return models
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return []

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def build_price_from_model(self, model_data: Dict) -> Optional[Dict]:
        """Build list_price and payout_price from model's pricing field."""
        pricing = model_data.get("pricing", {})
        if not pricing:
            return None
        try:
            input_price = str(Decimal(str(pricing["prompt"])) * 1_000_000)
            output_price = str(Decimal(str(pricing["completion"])) * 1_000_000)
            return {
                "description": "Pricing Per 1M Tokens Input/Output",
                "input": input_price,
                "output": output_price,
                "type": "one_million_tokens",
            }
        except (KeyError, Exception) as e:
            print(f"  ⚠️  Could not parse pricing: {e}")
            return None

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _render_template(self, template_name: str, context: Dict) -> str:
        template = self.jinja_env.get_template(template_name)
        return template.render(**context)

    def build_listing_context(self, model_id: str, price: Optional[Dict]) -> Dict:
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        return {
            "provider_name": PROVIDER_NAME,
            "offering_name": model_id,
            "env_api_key_name": ENV_API_KEY_NAME,
            "time_created": timestamp,
            "status": "ready",
            "list_price": price,
        }

    def build_offering_context(self, model_id: str, model_data: Dict, price: Optional[Dict]) -> Dict:
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        service_type = derive_service_type(model_id)

        details: Dict[str, Any] = {
            "model_name": model_id,
            "context_length": model_data.get("context_length"),
            "max_completion_tokens": model_data.get("max_completion_tokens"),
            "quantization": model_data.get("quantization"),
        }
        # Remove None values
        details = {k: v for k, v in details.items() if v is not None}

        return {
            "provider_name": PROVIDER_NAME,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "env_api_key_name": ENV_API_KEY_NAME,
            "time_created": timestamp,
            "offering_name": model_id,
            "display_name": model_id,
            "description": "",
            "service_type": service_type,
            "status": "ready",
            "api_base_url": self.api_base_url,
            "details": details,
            "payout_price": price,
        }

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _write_file(self, content: str, output_file: Path):
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✅ Written: {output_file}")
        except Exception as e:
            print(f"  ❌ Error writing {output_file}: {e}")

    def write_listing(self, model_id: str, price: Optional[Dict], output_dir: Path):
        context = self.build_listing_context(model_id, price)
        content = self._render_template("listing.json.j2", context)
        self._write_file(content, output_dir / "listing.json")

    def write_offering(self, model_id: str, model_data: Dict, price: Optional[Dict], output_dir: Path):
        context = self.build_offering_context(model_id, model_data, price)
        content = self._render_template("offering.json.j2", context)
        self._write_file(content, output_dir / "offering.json")

    def write_summary(self):
        print(f"   Total models: {self.summary['total_models']}")
        print(f"   Successful extractions: {self.summary['successful_extractions']}")
        print(f"   Skipped models: {self.summary['skipped_models']}")
        print(f"   Failed: {self.summary['failed_extractions']}")
        if self.summary["force_mode"]:
            print("   Force mode: Enabled")

    # ------------------------------------------------------------------
    # Deprecation
    # ------------------------------------------------------------------

    def mark_deprecated_services(self, output_dir: str, active_models: List[str], dry_run: bool = False):
        print("🔍 Checking for deprecated services...")
        base_path = Path(output_dir)
        if not base_path.exists():
            return
        active_dirs = {m.replace(":", "-") for m in active_models}
        deprecated_count = 0
        for item in base_path.iterdir():
            if not item.is_dir() or item.name in active_dirs:
                continue
            deprecated_count += 1
            print(f"  🗑️  Processing deprecated: {item.name}")
            for json_file in item.glob("*.json"):
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                    schema = data.get("schema", "")
                    updated = False
                    if schema == "offering_v1" and data.get("status") != "deprecated":
                        data["status"] = "deprecated"
                        updated = True
                    elif schema == "listing_v1" and data.get("status") != "upstream_deprecated":
                        data["status"] = "upstream_deprecated"
                        updated = True
                    if updated:
                        if dry_run:
                            print(f"    📝 [DRY-RUN] Would update {json_file.name}")
                        else:
                            with open(json_file, "w") as f:
                                json.dump(data, f, sort_keys=True, indent=2, separators=(",", ": "))
                                f.write("\n")
                            print(f"    ✅ Updated {json_file.name}")
                except Exception as e:
                    print(f"    ❌ Error: {e}")
        if deprecated_count == 0:
            print("  ✅ No deprecated services found")
        else:
            print(f"  🗑️  Processed {deprecated_count} deprecated services")

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def process_all_models(
        self,
        output_dir: str = "services",
        specific_models: Optional[List[str]] = None,
        force: bool = False,
        limit: Optional[int] = None,
        dry_run: bool = False,
    ):
        print("🚀 Starting CrofAI model extraction...\n")
        self.summary["force_mode"] = force
        self.summary["processing_limit"] = limit

        if dry_run:
            print("🔍 Dry-run mode — no files will be written")
        if force:
            print("💪 Force mode — existing files will be overwritten")

        if specific_models:
            print(f"🎯 Processing specific models: {', '.join(specific_models)}")
            models = [{"id": m} for m in specific_models]
            self.summary["total_models"] = len(models)
        else:
            models = self.get_all_models()
            if not models:
                print("❌ No models retrieved. Exiting.")
                return
            if force and limit is None:
                active_ids = [m.get("id", "").replace(":", "-") for m in models if m.get("id")]
                self.mark_deprecated_services(output_dir, active_ids, dry_run)

        skipped_count = 0
        processed_count = 0

        for i, model_data in enumerate(models, start=1):
            model_id = model_data.get("id", "").replace(":", "-")
            if not model_id:
                continue

            print(f"\n[{i}/{len(models)}] Processing: {model_id}")

            if limit and processed_count >= limit:
                print(f"🔢 Reached processing limit of {limit}, stopping...")
                break

            base_path = Path(output_dir)
            data_dir = base_path / model_id
            offering_file = data_dir / "offering.json"

            if not force and data_dir.exists() and offering_file.exists():
                print(f"  ⏭️  Skipping — files already exist (use --force to overwrite)")
                skipped_count += 1
                self.summary["skipped_models"] += 1
                continue

            processed_count += 1

            try:
                price = self.build_price_from_model(model_data)
                if price:
                    print(f"  💰 Pricing: input=${price['input']}, output=${price['output']}")

                if dry_run:
                    print(f"  📝 [DRY-RUN] Would write to {data_dir}")
                    self.summary["successful_extractions"] += 1
                    continue

                self.write_offering(model_id, model_data, price, data_dir)

                listing_file = data_dir / "listing.json"
                if listing_file.exists() and not force:
                    print("  ⏭️  Skipping existing listing.json (use --force to overwrite)")
                else:
                    self.write_listing(model_id, price, data_dir)

                self.summary["successful_extractions"] += 1
                print(f"  ✅ Successfully processed {model_id}")

            except Exception as e:
                print(f"  ❌ Error processing {model_id}: {e}")
                self.summary["failed_extractions"] += 1

        self.write_summary()
        print(f"\n🎉 Extraction complete! Check {output_dir}/ for results.")
        if skipped_count > 0:
            print(f"   ⏭️  Skipped {skipped_count} existing models (use --force to overwrite)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CrofAI model data")
    parser.add_argument("output_dir", nargs="?", default=str(Path(__file__).parent.parent / "services"))
    parser.add_argument("--models", nargs="+", help="Specific model IDs to process")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--limit", type=int, help="Limit number of models processed")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    args = parser.parse_args()

    api_key = os.environ.get("CROFAI_API_KEY", "")
    api_base_url = os.environ.get("CROFAI_API_BASE_URL", "https://ai.nahcrof.com/v2")

    if api_key:
        api_key = _sanitize_header_value(api_key)

    if not api_key:
        print("❌ Error: CROFAI_API_KEY environment variable not set.")
        sys.exit(1)

    script_dir = Path(__file__).parent
    templates_dir = script_dir.parent / "templates"

    if not templates_dir.exists():
        print(f"❌ Templates directory not found: {templates_dir}")
        sys.exit(1)

    extractor = CrofAIModelExtractor(api_key, api_base_url, templates_dir)
    extractor.process_all_models(
        args.output_dir,
        specific_models=args.models,
        force=args.force,
        limit=args.limit,
        dry_run=args.dry_run,
    )
