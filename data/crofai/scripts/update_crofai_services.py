#!/usr/bin/env python3
"""
pull_crofai_services.py / update_crofai_services.py - Extract model data from CROFAI API and pricing pages

This script mirrors the Fireworks version but targets CROFAI. It:
1. Retrieves models from the CROFAI API (or processes specific models if --models is used)
2. Attempts to extract pricing info from model pages (if model_base_url provided)
3. Gets detailed model information from API endpoints
4. Writes organized data to service.json and listing-svcreseller.json files
5. Marks deprecated services when models disappear (if --force used)
"""

import os
import sys
import json
import requests
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import re
from datetime import datetime
from urllib.parse import urljoin
from bs4 import BeautifulSoup

class CrofAIModelExtractor:
    def __init__(self, api_key: str, api_base_url: str, model_base_url: Optional[str] = None):
        self.api_key = api_key
        self.api_base_url = api_base_url.rstrip("/")
        # model_base_url optional; if not present we will attempt to infer it
        self.model_base_url = (model_base_url.rstrip("/") if model_base_url else None)
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (compatible; CrofAI-Service-Puller/1.0)"
        })
        self.extracted_data = {}
        self.summary = {
            "total_models": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "skipped_models": 0,
            "pricing_found": 0,
            "deprecated_models": [],
            "extraction_date": datetime.utcnow().isoformat() + "Z",
            "force_mode": False,
            "processing_limit": None,
        }

    def get_all_models(self) -> List[Dict]:
        """Retrieve all models from CROFAI API with simple pagination if available."""
        print("🔍 Fetching all models from CROFAI API...")

        # Common pattern: GET {api_base_url}/models or /v2/models
        url = f"{self.api_base_url}/models"
        all_models = []
        page_token = None
        page_count = 0
        max_pages = 100

        try:
            while page_count < max_pages:
                page_count += 1
                params = {"pageSize": 200}
                if page_token:
                    params["pageToken"] = page_token
                    print(f"📄 Fetching page {page_count} token...")

                print(f"📡 Requesting: {url} params={params}")
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                page_models = data.get("models", [])
                if not page_models:
                    # Some APIs return a top-level list instead of models field
                    if isinstance(data, list):
                        page_models = data
                    else:
                        print(f"⚠️  No models found on page {page_count}")
                        break

                all_models.extend(page_models)
                print(f"   Found {len(page_models)} models on page {page_count}")

                next_token = data.get("nextPageToken") or data.get("next_page_token") or None
                if not next_token:
                    print("✅ Pagination complete.")
                    break
                page_token = next_token

            self.summary["total_models"] = len(all_models)
            print(f"✅ Found {len(all_models)} models total across {page_count} pages")
            all_models.sort(key=lambda x: x.get("name", ""))
            return all_models

        except requests.RequestException as e:
            print(f"❌ Error fetching models: {e}")
            return []

    def _guess_model_base_url(self) -> Optional[str]:
        """Infer a model details page base URL when not provided."""
        if self.model_base_url:
            return self.model_base_url
        # try derive from api base (strip /api or /v2 and use site root)
        parsed = self.api_base_url
        # heuristics: replace '/v2' or '/api' with ''
        guessed = re.sub(r"/v\d+(/)?$", "", parsed)
        guessed = re.sub(r"/api(/)?$", "", guessed)
        guessed = guessed.rstrip("/") + "/models"
        print(f"ℹ️  Guessed model_base_url: {guessed}")
        return guessed

    def extract_pricing_from_page(self, model_name: str) -> Optional[Dict]:
        """Extract pricing info from the model page when available."""
        model_base = self._guess_model_base_url()
        if not model_base:
            return None

        clean_name = model_name.split("/")[-1].replace(":", "-")
        pricing_url = f"{model_base.rstrip('/')}/{clean_name}"
        print(f"  📄 Fetching pricing from: {pricing_url}")

        try:
            r = self.session.get(pricing_url, timeout=10)
            if r.status_code == 404:
                print(f"   ⚠️  Pricing page not found (404) for {model_name}")
                return None
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            pricing_info = {}

            # Look for currency-like text such as $0.XX
            text = soup.get_text(separator="\n")
            price_matches = re.findall(r"\$\s*\d+(\.\d+)?(?:\s*/\s*\$?\s*\d+(\.\d+)?)?", text)
            # price_matches returns tuples; check raw text search for price strings
            amount_search = re.search(r"\$\s*\d+(\.\d+)?(?:\s*/\s*\$?\s*\d+(\.\d+)?)?", text)
            if amount_search:
                price_str = amount_search.group(0).strip()
                pricing_info["unit"] = "unknown"
                pricing_info["price"] = price_str
                pricing_info["reference"] = pricing_url
                self.summary["pricing_found"] += 1
                print(f"  ✅ Extracted pricing: {price_str}")
                return pricing_info
            else:
                print("  ⚠️  No pricing data found on page")
                return None

        except requests.RequestException as e:
            print(f"  ❌ Error fetching pricing page: {e}")
            return None

    def get_model_details(self, model_name: str) -> Optional[Dict]:
        """Get detailed model info from API (try multiple endpoint formats)."""
        # Try: {api_base_url}/models/{id}
        model_id = model_name.split("/")[-1]
        endpoints = [
            f"{self.api_base_url}/models/{model_id}",
            f"{self.api_base_url}/model/{model_id}",
            f"{self.api_base_url}/accounts/crofai/models/{model_id}",
        ]
        for ep in endpoints:
            try:
                r = self.session.get(ep, timeout=10)
                if r.status_code == 200:
                    print(f"  ✅ Retrieved API details from {ep}")
                    return r.json()
                elif r.status_code == 404:
                    continue
                else:
                    r.raise_for_status()
            except requests.RequestException:
                continue
        print("  ⚠️  No API details available for model")
        return None

    def parse_pricing_string(self, price_string: str, pricing_unit: str) -> Dict[str, Any]:
        """Parse a price string into numeric values where possible."""
        pricing_data = {}
        try:
            # Handle "$0.12 / $0.34" style
            if "/" in price_string and "$" in price_string:
                parts = [p.strip().replace("$", "") for p in price_string.split("/")]
                nums = [float(re.sub(r"[^\d.]", "", p)) for p in parts if re.search(r"\d", p)]
                if len(nums) >= 2:
                    pricing_data["price_input"] = nums[0]
                    pricing_data["price_output"] = nums[1]
                elif len(nums) == 1:
                    pricing_data["price"] = nums[0]
            else:
                num_search = re.search(r"(\d+(\.\d+)?)", price_string)
                if num_search:
                    pricing_data["price"] = float(num_search.group(1))
        except Exception:
            pass
        return pricing_data

    def create_pricing_info_structure(self, pricing_data: Dict) -> List[Dict]:
        """Create structured pricing info compatible with downstream schema."""
        if not pricing_data:
            return []
        parsed = self.parse_pricing_string(pricing_data.get("price", "") if isinstance(pricing_data.get("price",""), str) else pricing_data.get("price",""), pricing_data.get("unit",""))
        if "price_input" in parsed and "price_output" in parsed:
            return [{
                "description": pricing_data.get("unit", "pricing"),
                "currency": "USD",
                "price_data": {"input": parsed["price_input"], "output": parsed["price_output"]},
                "unit": "one_million_tokens",
                "reference": pricing_data.get("reference")
            }]
        if "price" in parsed:
            return [{
                "description": pricing_data.get("unit", "pricing"),
                "currency": "USD",
                "price_data": {"price": parsed["price"]},
                "unit": "one_million_tokens",
                "reference": pricing_data.get("reference")
            }]
        # fallback: include raw string if parsing failed
        return [{
            "description": pricing_data.get("unit", "pricing"),
            "currency": "USD",
            "price_data": {"raw": pricing_data.get("price")},
            "unit": "unknown",
            "reference": pricing_data.get("reference")
        }]

    def determine_service_type(self, model_name: str, pricing_data: Optional[Dict]=None) -> str:
        """Heuristic to determine service type."""
        ml = model_name.lower()
        if pricing_data and "unit" in pricing_data:
            if "image" in pricing_data["unit"].lower():
                return "image_generation"
            if "token" in pricing_data["unit"].lower():
                if "embed" in ml or "embedding" in ml:
                    return "embedding"
                return "llm"
        # fallback guesses
        if any(k in ml for k in ["embed", "embedding"]):
            return "embedding"
        if any(k in ml for k in ["vision", "llava", "gpt-4-vision"]):
            return "vision_language_model"
        if any(k in ml for k in ["dalle", "stable", "midjourney", "flux"]):
            return "image_generation"
        return "llm"

    def _extract_model_size(self, model_name: str) -> Optional[str]:
        import re
        m = re.search(r"(\d+\.?\d*[bmk])", model_name.lower())
        return m.group(1) if m else None

    def create_service_data_structure(self, model_name: str, model_data: Dict, pricing_data: Optional[Dict], api_key: str) -> Dict:
        """Build service config (service.json) for a model."""
        now = datetime.utcnow()
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        handled_model_fields = ["createTime", "description", "displayName", "name", "public", "state", "updateTime"]
        header_fields = ["baseModelDetails"]
        base_mode_details_fields = ["checkpointFormat", "defaultPrecision", "modelType", "moe", "parameterCount", "supportsFireattention", "supportsMtp", "tunable", "worldSize"]
        top_level_model_fields = ["calibrated", "cluster", "contextLength", "conversationConfig", "defaultDraftModel", "defaultDraftTokenCount", "defaultSamplingParams", "deployedModelRefs", "deprecationDate", "fineTuningJob", "githubUrl", "huggingFaceUrl", "importedFrom", "kind", "peftDetails", "rlTunable", "status", "supportedPrecisions", "supportedPrecisionsWithCalibration", "supportsImageInput", "supportsLora", "supportsTools", "teftDetails", "trainingContextLength", "tunable", "useHfApplyChatTemplate"]

        service_config = {
            "schema": "service_v1",
            "time_created": model_data.get("createTime", timestamp),
            "name": model_name.split("/")[-1],
            "service_type": self.determine_service_type(model_name, pricing_data),
            "display_name": model_data.get("displayName", model_name.split("/")[-1]),
            "version": "",
            "description": model_data.get("description", ""),
            "upstream_status": (model_data.get("state") or "unknown").lower(),
            "details": {},
            "upstream_access_interface": {},
            "upstream_prices": {},
        }

        for field in top_level_model_fields:
            if field in model_data:
                service_config["details"][field] = model_data[field]

        if "baseModelDetails" in model_data:
            for field in base_mode_details_fields:
                if field in model_data["baseModelDetails"]:
                    service_config["details"][field] = model_data["baseModelDetails"][field]

        if pricing_data is not None:
            service_config["upstream_prices"] = self.create_pricing_info_structure(pricing_data)
        elif service_config["upstream_status"] == "ready":
            service_config["upstream_status"] = "uploading"

        service_config["upstream_access_interface"] = {
            "name": "CROFAI API",
            "api_key": api_key,
            "api_endpoint": f"{self.api_base_url}/inference",
            "access_method": "http",
            "rate_limits": [
                {"description": "Requests per minute", "limit": 60, "unit": "requests", "window": "minute"},
            ],
        }
        return service_config

    def create_operation_data_structure(self, pricing_data: Optional[Dict], upstream_ready: bool) -> Dict:
        now = datetime.utcnow()
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        operation_config = {
            "schema": "listing_v1",
            "seller_name": "svcreseller",
            "time_created": timestamp,
            "listing_status": "upstream_ready" if upstream_ready else "unknown",
            "user_access_interfaces": [],
            "user_prices": [],
        }
        if pricing_data is not None:
            operation_config["user_prices"] = self.create_pricing_info_structure(pricing_data)
        operation_config["user_access_interfaces"] = [
            {
                "name": "Provider API",
                "api_endpoint": f"{self.api_base_url}",
                "access_method": "http",
                "documents": [
                    {"title":"Python code example","description":"","mime_type":"python","category":"code_examples","file_path":"../../docs/code_example.py","is_active":True,"is_public":True},
                    {"title":"How to use this model","description":"","mime_type":"markdown","category":"getting_started","file_path":"../../docs/description.md","is_active":True,"is_public":True}
                ],
            }
        ]
        return operation_config

    def write_service_files(self, service_data, output_dir):
        base_path = Path(output_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        output_file = base_path / "service.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(service_data, f, sort_keys=True, indent=2, separators=(",", ": "))
                f.write("\n")
            print(f"  ✅ Written: {output_file}")
        except Exception as e:
            print(f"  ❌ Error writing {output_file}: {e}")

    def write_operation_files(self, operation_data, output_dir):
        base_path = Path(output_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        shared_path = base_path / ".." / ".." / "docs"
        shared_path.mkdir(parents=True, exist_ok=True)
        output_file = base_path / "listing-svcreseller.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(operation_data, f, sort_keys=True, indent=2, separators=(",", ": "))
                f.write("\n")
            print(f"  ✅ Written: {output_file}")
        except Exception as e:
            print(f"  ❌ Error writing {output_file}: {e}")

    def write_summary(self):
        try:
            print(f"   Total models: {self.summary['total_models']}")
            print(f"   Successful extractions: {self.summary['successful_extractions']}")
            print(f"   Skipped models: {self.summary['skipped_models']}")
            print(f"   With pricing data: {self.summary['pricing_found']}")
            print(f"   Deprecated models: {len(self.summary['deprecated_models'])}")
            if self.summary["force_mode"]:
                print("   Force mode: Enabled")
            if self.summary["processing_limit"]:
                print(f"   Processing limit: {self.summary['processing_limit']}")
        except Exception as e:
            print(f"❌ Error writing summary: {e}")

    def mark_deprecated_services(self, output_dir: str, active_models: List[str], dry_run: bool=False):
        print("🔍 Checking for deprecated services...")
        base_path = Path(output_dir)
        if not base_path.exists():
            print(f"  ⚠️  Output directory {output_dir} does not exist")
            return
        active_service_dirs = {m.split("/")[-1].replace(":", "_") for m in active_models}
        print(f"  Found {len(active_service_dirs)} active models")
        deprecated_count = 0
        for item in base_path.iterdir():
            if not item.is_dir():
                continue
            service_dir = item.name
            if service_dir in active_service_dirs:
                continue
            deprecated_count += 1
            print(f"  🗑️  Processing deprecated service: {service_dir}")
            for json_file in item.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    schema = data.get("schema")
                    updated = False
                    if schema == "service_v1":
                        current_status = data.get("upstream_status", "unknown")
                        if current_status != "deprecated":
                            data["upstream_status"] = "deprecated"
                            updated = True
                            status_msg = "service upstream_status to deprecated"
                    elif schema == "listing_v1":
                        current_op_status = data.get("listing_status", "unknown")
                        if current_op_status != "upstream_deprecated":
                            data["listing_status"] = "upstream_deprecated"
                            updated = True
                            status_msg = "operation operation_status to upstream_deprecated"
                    if updated:
                        if dry_run:
                            print(f"    📝 [DRY-RUN] Would update {json_file.name} {status_msg}")
                        else:
                            with open(json_file, "w", encoding="utf-8") as f:
                                json.dump(data, f, sort_keys=True, indent=2, separators=(",", ": "))
                                f.write("\n")
                            print(f"    ✅ Updated {json_file.name} {status_msg}")
                except Exception as e:
                    print(f"    ❌ Error updating {json_file}: {e}")
        if deprecated_count == 0:
            print("  ✅ No deprecated services found")
        else:
            print(f"  🗑️  Processed {deprecated_count} deprecated services")

    def process_all_models(self, output_dir: str = "services", specific_models: Optional[List[str]] = None, force: bool = False, limit: Optional[int] = None, dry_run: bool=False):
        print("🚀 Starting CROFAI model extraction...\n")
        self.summary["force_mode"] = force
        self.summary["processing_limit"] = limit
        if dry_run:
            print("🔍 Dry-run mode enabled")
        if force:
            print("💪 Force mode enabled")
        if limit:
            print(f"🔢 Processing limit set to {limit}")
        if specific_models:
            print(f"🎯 Processing specific models: {', '.join(specific_models)}")
            models = [{"name": m} for m in specific_models]
            self.summary["total_models"] = len(models)
        else:
            models = self.get_all_models()
            if not models:
                print("❌ No models retrieved. Exiting.")
                return
            if force and limit is None:
                active_model_names = [m.get("name","") for m in models if m.get("name")]
                self.mark_deprecated_services(output_dir, active_model_names, dry_run)

        skipped_count = 0
        processed_count = 0
        for i, model_data in enumerate(models, start=1):
            model_name = model_data.get("name","")
            if not model_name:
                continue
            print(f"\n[{i}/{len(models)}] Processing: {model_name}")
            if limit and processed_count >= limit:
                print(f"🔢 Reached processing limit of {limit} models, stopping...")
                break
            base_path = Path(output_dir)
            dir_name = model_name.split("/")[-1].replace(":", "_")
            data_dir = base_path / dir_name
            data_file = data_dir / "service.json"
            if not force and data_dir.exists() and data_file.exists():
                print(f"  ⏭️  Skipping {model_name} - service file already exists (use --force to overwrite)")
                skipped_count += 1
                self.summary["skipped_models"] += 1
                continue
            processed_count += 1
            try:
                try:
                    pricing_data = self.extract_pricing_from_page(model_name)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"  ❌ Error parsing pricing page: {e}")
                    pricing_data = None

                details = self.get_model_details(model_name) or {}
                # Merge API details into model_data
                if isinstance(model_data, dict):
                    merged = dict(model_data)
                    merged.update(details)
                else:
                    merged = details

                service_config = self.create_service_data_structure(model_name, merged, pricing_data, self.api_key)
                operation_config = self.create_operation_data_structure(pricing_data, service_config["upstream_status"] == "ready")
                print("  📝 Generated service data")
                self.extracted_data[model_name] = service_config
                self.summary["successful_extractions"] += 1
                if dry_run:
                    print(f"  📝 [DRY-RUN] Would write service files to {data_dir}")
                else:
                    print(f"  📝 Writing service files to {data_dir}...")
                    self.write_service_files(service_config, data_dir)
                if (data_dir / "listing-svcreseller.json").exists():
                    print("  ⚠️  Ignore existing listing-svcreseller.json. --force will not help. Please remove if you would like to re-generate listing-svcreseller.json file.")
                else:
                    if dry_run:
                        print(f"  📝 [DRY-RUN] Would write listing files to {data_dir}")
                    else:
                        print(f"  📝 Writing listing files to {data_dir}...")
                        self.write_operation_files(operation_config, data_dir)
                print(f"  ✅ Successfully processed {model_name}")
            except Exception as e:
                print(f"  ❌ Error processing {model_name}: {e}")
                self.summary["failed_extractions"] += 1

        self.write_summary()
        print(f"\n🎉 Extraction complete! Check {output_dir}/ for results.")
        if skipped_count > 0:
            print(f"   ⏭️  Skipped {skipped_count} existing models (use --force to overwrite)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract model data from CROFAI API and pricing pages")
    parser.add_argument("output_dir", nargs="?", default="services", help="Output directory for service files (default: services)")
    parser.add_argument("--models", nargs="+", help="Specific model names to process (e.g., --models accounts/crofai/my-model)")
    parser.add_argument("--force", action="store_true", help="Force update existing service directories. Without this flag, existing directories will be skipped.")
    parser.add_argument("--limit", type=int, help="Limit the number of models to process.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually writing files.")
    args = parser.parse_args()

    api_key = os.environ.get("CROFAI_API_KEY")
    api_base_url = os.environ.get("CROFAI_API_BASE_URL")
    model_base_url = os.environ.get("CROFAI_MODEL_BASE_URL", None)

    if not api_key:
        print("❌ Error: No API key provided. Set CROFAI_API_KEY environment variable.")
        sys.exit(1)
    if not api_base_url:
        print("❌ Error: No API base URL provided. Set CROFAI_API_BASE_URL environment variable.")
        sys.exit(1)

    extractor = CrofAIModelExtractor(api_key, api_base_url, model_base_url)
    extractor.process_all_models(args.output_dir, specific_models=args.models, force=args.force, limit=args.limit, dry_run=args.dry_run)
