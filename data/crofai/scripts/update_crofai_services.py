#!/usr/bin/env python3
"""
CROFAI Service Extractor
Pulls models from CROFAI API, fetches pricing, writes service.json and listing-svcreseller.json
"""

import os
import sys
import json
import requests
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import re
from datetime import datetime
from bs4 import BeautifulSoup

class CrofAIModelExtractor:
    def __init__(self, api_key: str, api_base_url: str):
        self.api_key = api_key
        self.api_base_url = api_base_url.rstrip("/")
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
        """Fetch all models from CROFAI API."""
        url = f"{self.api_base_url}/models"
        all_models = []
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data") or data.get("models") or []
            all_models.extend(models)
            self.summary["total_models"] = len(all_models)
            all_models.sort(key=lambda x: x.get("name", ""))
            print(f"✅ Retrieved {len(all_models)} models from API")
            return all_models
        except Exception as e:
            print(f"❌ Error fetching models: {e}")
            return []

    def extract_pricing_from_page(self, model_name: str) -> Optional[Dict]:
        """Fetch pricing from the CROFAI pricing page."""
        pricing_url = "https://ai.nahcrof.com/pricing"
        print(f"  📄 Fetching pricing from: {pricing_url}")

        try:
            r = self.session.get(pricing_url, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator="\n")

            pattern = re.compile(rf"{re.escape(model_name)}.*?(\$\s*\d+(\.\d+)?)", re.IGNORECASE)
            match = pattern.search(text)
            if match:
                price_str = match.group(1).strip()
                self.summary["pricing_found"] += 1
                print(f"  ✅ Extracted pricing for {model_name}: {price_str}")
                return {"price": price_str, "unit": "one_million_tokens", "reference": pricing_url}
            print(f"  ⚠️  No pricing found for {model_name} on page")
            return None
        except Exception as e:
            print(f"  ❌ Error fetching pricing page: {e}")
            return None

    def get_model_details(self, model_name: str) -> Optional[Dict]:
        """Fetch detailed info from API."""
        model_id = model_name.split("/")[-1]
        endpoints = [
            f"{self.api_base_url}/models/{model_id}",
            f"{self.api_base_url}/model/{model_id}",
        ]
        for ep in endpoints:
            try:
                r = self.session.get(ep, timeout=10)
                if r.status_code == 200:
                    return r.json()
            except:
                continue
        return {}

    def parse_pricing_string(self, price_string: str) -> Dict[str, float]:
        nums = re.findall(r"\d+(\.\d+)?", price_string)
        if nums:
            return {"price": float(nums[0])}
        return {}

    def create_pricing_info_structure(self, pricing_data: Optional[Dict]) -> List[Dict]:
        if not pricing_data:
            return []
        parsed = self.parse_pricing_string(pricing_data.get("price", ""))
        if "price" in parsed:
            return [{
                "description": pricing_data.get("unit", "pricing"),
                "currency": "USD",
                "price_data": {"price": parsed["price"]},
                "unit": "one_million_tokens",
                "reference": pricing_data.get("reference")
            }]
        return []

    def determine_service_type(self, model_name: str, pricing_data: Optional[Dict]=None) -> str:
        ml = model_name.lower()
        if pricing_data and "unit" in pricing_data and "token" in pricing_data["unit"].lower():
            if "embed" in ml:
                return "embedding"
            return "llm"
        if any(k in ml for k in ["embed", "embedding"]):
            return "embedding"
        if any(k in ml for k in ["dalle", "stable", "midjourney"]):
            return "image_generation"
        return "llm"

    def create_service_data_structure(self, model_name: str, model_data: Dict, pricing_data: Optional[Dict], api_key: str) -> Dict:
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        service_config = {
            "schema": "service_v1",
            "time_created": model_data.get("createTime", timestamp),
            "name": model_name.split("/")[-1],
            "service_type": self.determine_service_type(model_name, pricing_data),
            "display_name": model_data.get("displayName", model_name.split("/")[-1]),
            "description": model_data.get("description", ""),
            "upstream_status": (model_data.get("state") or "unknown").lower(),
            "details": model_data,
            "upstream_prices": self.create_pricing_info_structure(pricing_data),
            "upstream_access_interface": {
                "name": "CROFAI API",
                "api_key": api_key,
                "api_endpoint": f"{self.api_base_url}/inference",
                "access_method": "http",
            }
        }
        return service_config

    def create_operation_data_structure(self, pricing_data: Optional[Dict], upstream_ready: bool) -> Dict:
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        op_config = {
            "schema": "listing_v1",
            "seller_name": "svcreseller",
            "time_created": timestamp,
            "listing_status": "upstream_ready" if upstream_ready else "unknown",
            "user_access_interfaces": [{"name": "Provider API", "api_endpoint": self.api_base_url, "access_method": "http"}],
            "user_prices": self.create_pricing_info_structure(pricing_data) if pricing_data else [],
        }
        return op_config

    def write_service_files(self, service_data, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "service.json", "w", encoding="utf-8") as f:
            json.dump(service_data, f, indent=2)
        with open(output_dir / "listing-svcreseller.json", "w", encoding="utf-8") as f:
            json.dump(self.create_operation_data_structure(service_data.get("upstream_prices"), service_data.get("upstream_status")=="ready"), f, indent=2)
        print(f"  ✅ Written service.json & listing-svcreseller.json in {output_dir}")

    def mark_deprecated_services(self, output_dir: str, active_models: List[str], dry_run: bool=False):
        print("🔍 Checking for deprecated services...")
        base_path = Path(output_dir)
        active_set = {m.split("/")[-1].replace(":", "_").replace(" ", "_") for m in active_models}
        for item in base_path.iterdir():
            if item.is_dir() and item.name not in active_set:
                for f in item.glob("*.json"):
                    if dry_run:
                        print(f"  📝 [DRY-RUN] Would mark {f.name} as deprecated")
                    else:
                        data = json.loads(f.read_text())
                        if data.get("schema") == "service_v1":
                            data["upstream_status"] = "deprecated"
                        elif data.get("schema") == "listing_v1":
                            data["listing_status"] = "upstream_deprecated"
                        f.write_text(json.dumps(data, indent=2))
                        print(f"  ✅ Marked {f.name} as deprecated")

    def process_all_models(self, output_dir: str = "crofai/services", specific_models: Optional[List[str]] = None, force: bool=False, dry_run: bool=False):
        self.summary["force_mode"] = force
        if specific_models:
            models = [{"name": m} for m in specific_models]
        else:
            models = self.get_all_models()
            active_model_names = [m.get("name","") for m in models]
            if force:
                self.mark_deprecated_services(output_dir, active_model_names, dry_run=dry_run)

        for i, model_data in enumerate(models, 1):
            model_name = model_data.get("name")
            if not model_name:
                continue
            print(f"\n[{i}/{len(models)}] Processing: {model_name}")
            dir_name = model_name.split("/")[-1].replace(":", "_").replace(" ", "_")
            data_dir = Path(output_dir) / dir_name

            if not force and (data_dir / "service.json").exists():
                print(f"  ⏭️  Skipping {model_name} (service.json exists)")
                self.summary["skipped_models"] += 1
                continue

            pricing_data = self.extract_pricing_from_page(model_name)
            details = self.get_model_details(model_name)
            merged_data = {**model_data, **details}
            service_config = self.create_service_data_structure(model_name, merged_data, pricing_data, self.api_key)

            if dry_run:
                print(f"  📝 [DRY-RUN] Would write service.json & listing-svcreseller.json for {model_name}")
            else:
                self.write_service_files(service_config, data_dir)
                self.summary["successful_extractions"] += 1

        print(f"\n🎉 Extraction complete! {self.summary['successful_extractions']} models processed.")
        if dry_run:
            print("⚠️  Dry-run mode enabled, no files were written.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CROFAI services and pricing")
    parser.add_argument("output_dir", nargs="?", default="crofai/services")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--force", action="store_true", help="Overwrite existing services and mark deprecated")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing files")
    args = parser.parse_args()

    api_key = os.environ.get("CROFAI_API_KEY")
    api_base_url = os.environ.get("CROFAI_API_BASE_URL")
    if not api_key or not api_base_url:
        print("❌ Set CROFAI_API_KEY and CROFAI_API_BASE_URL environment variables.")
        sys.exit(1)

    extractor = CrofAIModelExtractor(api_key, api_base_url)
    extractor.process_all_models(args.output_dir, specific_models=args.models, force=args.force, dry_run=args.dry_run)
