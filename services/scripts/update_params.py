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
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from jinja2 import Environment, FileSystemLoader, StrictUndefined

# Canonical metadata helpers shared with deepseek / fireworks / etc. Resolves
# context_length and parameter_count from the OpenRouter -> LiteLLM -> HuggingFace
# fallback chain when CrofAI's API doesn't supply them (it never returns
# parameter_count; context_length is sometimes missing).
from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup
from unitysvc_sellers.model_overrides import load_model_overrides
from unitysvc_sellers.params_render import write_params_from_iterator

#: The repo's committed corrections to fetched metadata — skip/deprecate flags
#: and template-var overrides in services/model_overrides.toml, re-applied on
#: every populate run (see unitysvc-sellers docs/service-templates.md).
SERVICES_DIR = Path(__file__).resolve().parent.parent


PROVIDER_NAME = "crofai"

PROVIDER_DISPLAY_NAME = "CrofAI"
ENV_API_KEY_NAME = "CROFAI_API_KEY"

# What the platform adds on top of the upstream rate for the MANAGED channel,
# where UnitySVC's own key pays the provider and the customer pays UnitySVC.
# The byok channel is unaffected: the customer's key pays the provider directly,
# so there is nothing to mark up and nothing to pay out.
#
# This is the seller's own list price, computed here at populate time — not a
# platform-side calculation. `list_price` is stored already marked up, so what
# is displayed is exactly what is billed.
PLATFORM_MARKUP = Decimal("1.15")

# Rounding for the marked-up rate. 3dp, chosen by measuring the effective markup
# across every upstream rate this catalog carries ($0.04 - $10):
#
#     3dp  ->  15.0% - 15.7%   (drift under a percentage point)
#     2dp  ->  13.3% - 25.0%   ($0.04 becomes $0.05, a quarter markup)
#     2sf  ->  10.0% - 20.0%   ($1 becomes $1.10, $10 becomes $12)
#
# The exact percentage is not the point — landing near it everywhere is. 2dp
# looks tidier but distorts cheap models badly, and cheap models are most of
# this catalog. `_fmt_price` drops trailing zeros afterwards, so a rate that
# needs no third decimal does not show one.
PRICE_PLACES = Decimal("0.001")


def _now_iso() -> str:
    """UTC timestamp matching the platform's millisecond-Z format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _fmt_price(value) -> str:
    """Per-1M price as a compact string, dropping trailing zeros ("0.150" -> "0.15",
    "10.000000" -> "10"). format(..., 'f') expands any exponent normalize() produces."""
    return format(Decimal(str(value)).normalize(), "f")


def _as_positive_int(value) -> Optional[int]:
    """Coerce ``value`` to a positive ``int`` or ``None``.

    The platform validator (unitysvc#863) rejects strings, zero, negative
    values, and bool. CrofAI usually returns ``context_length`` as an int
    already, but be defensive — same shape fireworks uses.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


# CrofAI uses bare lowercased ids like ``deepseek-v3.2`` / ``glm-5`` /
# ``kimi-k2.5`` / ``qwen3.5-397b-a17b``. The canonical resolver
# (``ModelDataLookup.get_canonical_metadata``) keys ``parameter_count`` by
# the HuggingFace repo path (``org/Repo-Name``) via the safetensors
# metadata fetcher, so a bare CrofAI id never matches and we get
# ``parameter_count=None`` for every model.
#
# This table maps known CrofAI prefixes to a function that produces the
# canonical HF id. ``apply_hf_id`` returns ``None`` when nothing matches —
# in that case we just pass the bare id through (which still resolves
# ``context_length`` from OpenRouter / LiteLLM, just not parameter_count).
def _to_hf_id(model_id: str) -> Optional[str]:
    mid = model_id
    # deepseek-v3.2  -> deepseek-ai/DeepSeek-V3.2
    # deepseek-v4-pro -> deepseek-ai/DeepSeek-V4-Pro
    if mid.startswith("deepseek-"):
        rest = mid[len("deepseek-"):]
        return "deepseek-ai/DeepSeek-" + "-".join(p.capitalize() for p in rest.split("-"))
    # glm-5 -> zai-org/GLM-5;  glm-4.7-flash -> zai-org/GLM-4.7-Flash
    if mid.startswith("glm-"):
        rest = mid[len("glm-"):]
        return "zai-org/GLM-" + "-".join(p.capitalize() for p in rest.split("-"))
    # kimi-k2.5 -> moonshotai/Kimi-K2.5;  kimi-k2.6-precision -> moonshotai/Kimi-K2.6-Precision
    if mid.startswith("kimi-"):
        rest = mid[len("kimi-"):]
        return "moonshotai/Kimi-" + "-".join(p.capitalize() for p in rest.split("-"))
    # qwen3.5-397b-a17b -> Qwen/Qwen3.5-397B-A17B;  qwen3.6-27b -> Qwen/Qwen3.6-27B
    # The leading ``qwen<ver>`` segment becomes title-cased ``Qwen<ver>``
    # (HF convention); subsequent segments containing digits become
    # all-uppercase (``397B``, ``A17B``).
    if mid.startswith("qwen"):
        parts = mid.split("-")
        head = "Qwen" + parts[0][len("qwen"):]  # e.g. "Qwen3.5" / "Qwen3.6"
        tail = [p.upper() if any(c.isdigit() for c in p) else p.capitalize() for p in parts[1:]]
        return "Qwen/" + "-".join([head, *tail])
    # gemma-4-31b-it -> google/gemma-4-31b-it (HF preserves case here)
    if mid.startswith("gemma-"):
        return "google/" + mid
    # minimax-m2.5 -> MiniMaxAI/MiniMax-M2.5
    if mid.startswith("minimax-"):
        rest = mid[len("minimax-"):]
        return "MiniMaxAI/MiniMax-" + "-".join(p.upper() if p[0].isalpha() and any(c.isdigit() for c in p) else p.capitalize() for p in rest.split("-"))
    return None


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
            "extraction_date": datetime.now().isoformat(),
            "processing_limit": None,
        }

        self.jinja_env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        self.jinja_env.filters["tojson"] = lambda v: json.dumps(v)

        # Committed corrections (services/model_overrides.toml) — fail-loud on
        # a malformed file so a typo'd entry can't silently re-break its fix.
        self.overrides = load_model_overrides(SERVICES_DIR)

        # Lazy-init: only fetch canonical model data on first lookup so dry
        # runs / --models filter passes don't pay the network cost upfront.
        self._fetcher: Optional[ModelDataFetcher] = None

    def _canonical_metadata(self) -> ModelDataFetcher:
        if self._fetcher is None:
            self._fetcher = ModelDataFetcher()
        return self._fetcher

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
            listed = len(models)
            models = [m for m in models if not self.overrides.skip(m.get("id", ""))]
            if len(models) != listed:
                print(f"⚠️  Skipped {listed - len(models)} model(s) via model_overrides.toml")
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
        """Upstream and marked-up rates for one model.

        Returns ``{"upstream": {...}, "managed": {...}}`` — two ``one_million_tokens``
        prices, NOT one shared object. They mean different things and must never
        be the same dict:

        * ``upstream`` is what CrofAI charges. It becomes ``payout_price`` on the
          managed channel: what the platform owes the seller, which does not move
          when we change what we charge.
        * ``managed`` is upstream × ``PLATFORM_MARKUP``. It becomes ``list_price``
          on the managed channel: what the customer pays UnitySVC.

        Deriving one from the other here, rather than storing a percentage, is
        what keeps them consistent: a ``revenue_share`` payout would track the
        list price and so would silently follow any change to the markup, an
        override, or a promotion — see unitysvc/unitysvc#1892.
        """
        pricing = model_data.get("pricing", {})
        if not pricing:
            return None
        try:
            # CrofAI's API already returns prices PER 1M TOKENS (e.g. prompt
            # "0.35"), so use them as-is. The previous ``* 1_000_000`` inflated
            # every managed price a million-fold — a real billing bug on the paid
            # managed channel, not just a display glitch.
            up_in = Decimal(str(pricing["prompt"]))
            up_out = Decimal(str(pricing["completion"]))
            # Quantize after marking up, so the stored rate is exactly what is
            # billed rather than a repeating decimal truncated at render time.
            mk_in = (up_in * PLATFORM_MARKUP).quantize(PRICE_PLACES, rounding=ROUND_HALF_UP)
            mk_out = (up_out * PLATFORM_MARKUP).quantize(PRICE_PLACES, rounding=ROUND_HALF_UP)
            return {
                "upstream": {
                    "input": _fmt_price(up_in),
                    "output": _fmt_price(up_out),
                    "type": "one_million_tokens",
                },
                "managed": {
                    "description": (
                        f"${_fmt_price(mk_in)}/${_fmt_price(mk_out)}"
                        f" / 1M input/output tokens"
                    ),
                    "input": _fmt_price(mk_in),
                    "output": _fmt_price(mk_out),
                    "type": "one_million_tokens",
                },
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

    def build_listing_context(self, model_id: str, price: Optional[Dict], time_created: Optional[str] = None) -> Dict:
        return {
            "provider_name": PROVIDER_NAME,
            "offering_name": model_id,
            "env_api_key_name": ENV_API_KEY_NAME,
            "time_created": time_created or _now_iso(),
            "status": "ready",
            # The MARKED-UP managed rate. The listing template wraps this into
            # the channel price (byok free, managed metered).
            "list_price": (price or {}).get("managed"),
        }

    def build_offering_context(
        self, model_id: str, model_data: Dict, price: Optional[Dict], time_created: Optional[str] = None
    ) -> Dict:
        timestamp = time_created or _now_iso()
        service_type = derive_service_type(model_id)

        details: Dict[str, Any] = {
            "model_name": model_id,
            "max_completion_tokens": model_data.get("max_completion_tokens"),
            "quantization": model_data.get("quantization"),
        }
        # Drop None-valued upstream fields so the rendered offering is clean.
        details = {k: v for k, v in details.items() if v is not None}

        # Canonical (snake_case) metadata required by the platform validator
        # for LLM offerings. Both keys must be present; null asserts "unknown".
        # CrofAI's API gives us context_length but never parameter_count, so
        # ask the canonical helper (OpenRouter -> LiteLLM -> HuggingFace) to
        # fill in whatever it can. metadata_sources records provenance.
        #
        # The HuggingFace safetensors fetcher (which is the only source for
        # parameter_count) keys by HF repo path — so for known model
        # families we lift the bare CrofAI id to its HF form before looking
        # up. Bare lookup still resolves context_length via OpenRouter.
        hf_id = _to_hf_id(model_id)
        canonical = ModelDataLookup.get_canonical_metadata(
            hf_id or model_id, fetcher=self._canonical_metadata()
        )
        # Prefer CrofAI's context_length when present; otherwise use the
        # canonical resolver's answer. parameter_count comes solely from the
        # canonical resolver — CrofAI doesn't surface it.
        upstream_ctx = _as_positive_int(model_data.get("context_length"))
        details["context_length"] = upstream_ctx if upstream_ctx is not None else canonical["context_length"]
        details["parameter_count"] = canonical["parameter_count"]
        sources = {k: v for k, v in (canonical.get("sources") or {}).items() if v}
        # Provenance: only record sources we actually used.
        used_sources: Dict[str, str] = {}
        if upstream_ctx is None and canonical["context_length"] is not None and "context_length" in sources:
            used_sources["context_length"] = sources["context_length"]
        if canonical["parameter_count"] is not None and "parameter_count" in sources:
            used_sources["parameter_count"] = sources["parameter_count"]
        if used_sources:
            details["metadata_sources"] = used_sources

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
            # The UPSTREAM rate, deliberately not the list price: this is what
            # the platform owes the seller, and it must not move when the markup
            # or a promotion moves. The offering template wraps it into the
            # channel payout (byok free, managed upstream).
            "payout_price": (price or {}).get("upstream"),
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

    @staticmethod
    def _existing_time_created(path: Path) -> Optional[str]:
        """Return the time_created already recorded in a spec file, if any, so
        regenerating an unchanged service is idempotent (no daily churn)."""
        if path.exists():
            try:
                return json.loads(path.read_text()).get("time_created")
            except (json.JSONDecodeError, OSError):
                return None
        return None

    @staticmethod
    def _param_time_created(path: Path) -> Optional[str]:
        """time_created recorded inside a committed param file's ``parameters``
        (the post-migration home of the field), so re-runs stay idempotent."""
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return (data.get("parameters") or {}).get("time_created")
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def write_listing(self, model_id: str, price: Optional[Dict], output_dir: Path):
        created = self._existing_time_created(output_dir / "listing.json")
        context = self.build_listing_context(model_id, price, time_created=created)
        content = self._render_template("listing.json.j2", context)
        self._write_file(content, output_dir / "listing.json")

    def write_offering(self, model_id: str, model_data: Dict, price: Optional[Dict], output_dir: Path):
        created = self._existing_time_created(output_dir / "offering.json")
        context = self.build_offering_context(model_id, model_data, price, time_created=created)
        content = self._render_template("offering.json.j2", context)
        self._write_file(content, output_dir / "offering.json")

    def write_provider(self, output_dir: Path):
        """Copy the static templates/provider.json into the service folder so
        each folder is self-contained."""
        prov = json.loads((self.templates_dir / "provider.json").read_text())
        content = json.dumps(prov, sort_keys=True, indent=2) + "\n"
        self._write_file(content, output_dir / "provider.json")

    def write_summary(self):
        print(f"   Total models: {self.summary['total_models']}")
        print(f"   Successful extractions: {self.summary['successful_extractions']}")
        print(f"   Failed: {self.summary['failed_extractions']}")

    # ------------------------------------------------------------------
    # Deprecation
    # ------------------------------------------------------------------

    def mark_deprecated_services(self, output_dir: str, active_models: List[str], dry_run: bool = False):
        """Set ``parameters.status = "deprecated"`` on the param file of every
        local service whose model is no longer offered upstream.

        Param-file layout (#1263): services are ``<output_dir>/crofai/<model>.json``
        compact param files, NOT expanded ``<model>/offering.json`` folders — the
        previous folder-globbing version of this function silently matched
        nothing after the layout migration, which is how delisted models
        lingered in the catalog until they were rejected on staging.
        """
        print("🔍 Checking for deprecated services...")
        base_path = Path(output_dir) / PROVIDER_NAME
        if not base_path.exists():
            return
        active = {m.replace(":", "-") for m in active_models}
        deprecated_count = 0
        for param_file in sorted(base_path.rglob("*.json")):
            if param_file.name.endswith(".service.json"):
                continue
            model_id = param_file.relative_to(base_path).as_posix()[: -len(".json")]
            if model_id in active:
                continue
            try:
                data = json.loads(param_file.read_text())
            except Exception as e:
                print(f"    ❌ Error reading {param_file.name}: {e}")
                continue
            params = data.get("parameters", {})
            if params.get("status") == "deprecated":
                continue
            deprecated_count += 1
            if dry_run:
                print(f"  📝 [DRY-RUN] Would deprecate: {model_id}")
                continue
            params["status"] = "deprecated"
            data["parameters"] = params
            with open(param_file, "w") as f:
                json.dump(data, f, sort_keys=True, indent=2, separators=(",", ": "))
                f.write("\n")
            print(f"  🗑️  Deprecated: {model_id}")
        if deprecated_count == 0:
            print("  ✅ No deprecated services found")
        else:
            print(f"  🗑️  Processed {deprecated_count} deprecated services")

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def process_all_models(
        self,
        output_dir: str = "specs",
        specific_models: Optional[List[str]] = None,
        limit: Optional[int] = None,
        dry_run: bool = False,
    ):
        print("🚀 Starting CrofAI model extraction...\n")
        self.summary["processing_limit"] = limit

        if dry_run:
            print("🔍 Dry-run mode — no files will be written")

        if specific_models:
            print(f"🎯 Processing specific models: {', '.join(specific_models)}")
            models = [{"id": m} for m in specific_models]
            self.summary["total_models"] = len(models)
        else:
            models = self.get_all_models()
            if not models:
                print("❌ No models retrieved. Exiting.")
                return
            # Full sync: deprecate local services no longer offered upstream.
            # (Skipped when --limit truncates the model list.)
            if limit is None:
                active_ids = [m.get("id", "").replace(":", "-") for m in models if m.get("id")]
                self.mark_deprecated_services(output_dir, active_ids, dry_run)
                # Flag override entries that no longer match anything — a
                # stale entry usually means the model was renamed or finally
                # dropped upstream and the override can be retired.
                cataloged = {
                    p.relative_to(Path(output_dir) / PROVIDER_NAME).as_posix()[: -len(".json")]
                    for p in (Path(output_dir) / PROVIDER_NAME).rglob("*.json")
                    if not p.name.endswith(".service.json")
                }
                self.overrides.warn_unmatched(set(active_ids) | cataloged)

        processed_count = 0

        param_contexts: List[Dict] = []

        for i, model_data in enumerate(models, start=1):
            model_id = model_data.get("id", "").replace(":", "-")
            if not model_id:
                continue

            print(f"\n[{i}/{len(models)}] Processing: {model_id}")

            if limit and processed_count >= limit:
                print(f"🔢 Reached processing limit of {limit}, stopping...")
                break

            processed_count += 1

            try:
                price = self.build_price_from_model(model_data)
                if price:
                    up, mg = price["upstream"], price["managed"]
                    print(
                        f"  💰 upstream in/out ${up['input']}/${up['output']}"
                        f"  →  managed (×{PLATFORM_MARKUP}) ${mg['input']}/${mg['output']}"
                    )

                if dry_run:
                    print(f"  📝 [DRY-RUN] Would write param file for {PROVIDER_NAME}/{model_id}")
                    self.summary["successful_extractions"] += 1
                    continue

                # Preserve time_created so unchanged services produce no churn:
                # prefer the committed param file, fall back to the legacy
                # expanded offering.json (first run after the param migration).
                base = Path(output_dir) / PROVIDER_NAME
                created = self._param_time_created(base / f"{model_id}.json") or self._existing_time_created(
                    base / model_id / "offering.json"
                )

                # Merge the offering + listing render contexts into one param
                # context; name = listing.name = the param file's path.
                offering = self.build_offering_context(model_id, model_data, price, time_created=created)
                listing = self.build_listing_context(model_id, price, time_created=created)
                ctx = self.overrides.apply(
                    model_id, {**offering, **listing, "name": f"{PROVIDER_NAME}/{model_id}"}
                )
                param_contexts.append(ctx)

                self.summary["successful_extractions"] += 1
                print(f"  ✅ Successfully processed {model_id}")

            except Exception as e:
                print(f"  ❌ Error processing {model_id}: {e}")
                self.summary["failed_extractions"] += 1

        if not dry_run:
            write_params_from_iterator(iter(param_contexts), output_dir)

        self.write_summary()
        print(f"\n🎉 Extraction complete! Check {output_dir}/ for results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CrofAI model data")
    parser.add_argument("output_dir", nargs="?", default=str(Path(__file__).parent.parent / "specs"))
    parser.add_argument("--models", nargs="+", help="Specific model IDs to process")
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
        limit=args.limit,
        dry_run=args.dry_run,
    )
