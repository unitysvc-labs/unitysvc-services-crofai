# unitysvc-services-crofai

CrofAI service data for the UnitySVC marketplace. The data here is consumed
by the platform's data-validation, populate-services, and upload workflows
to keep the live catalog in sync with the provider.

[CrofAI homepage](https://nahcrof.com/) — OpenAI-compatible chat-completion
gateway hosting Llama, DeepSeek, Qwen, GLM, Kimi, MiniMax, and Gemma family
models.

## Repository layout

```
data/crofai/
├── provider.toml                   # Provider metadata + populate-services config
├── README.md                       # Provider-side notes
├── scripts/update_services.py      # Pulls model list + pricing from CrofAI's API
└── services/                       # One directory per service offering
    └── <model-id>/
        ├── offering.json           # Provider-side technical spec
        ├── listing.json            # Customer-facing listing (uses $doc_preset
        │                           #   references for code examples + connectivity)
        └── listing.override.json   # Set by upload-data workflow on first publish
```

Documents (connectivity test, description, code examples) are pulled from
the centralised preset library in
[`unitysvc-data`](https://github.com/unitysvc/unitysvc-data) so every CrofAI
listing renders identically to every other LLM listing on the marketplace.
The listings reference these presets by name:

- `llm_connectivity` — POST `/chat/completions` smoke test
- `llm_description` — generic LLM description / best practices
- `llm_code_example_javascript` — `fetch`-based example
- `llm_code_example_requests` — Python `requests` example
- `llm_code_example_fc_requests` — Python function-calling example
- `llm_code_example_shell` — `curl` example

## Local development

```bash
pip install unitysvc-sellers
```

```bash
# Validate the data files
usvc_seller data validate

# Format JSON / TOML
usvc_seller data format

# Run code-example tests against the upstream API
#   (requires CROFAI_API_KEY in the environment)
usvc_seller data run-tests

# Pull the latest model list from CrofAI's API and refresh listings
#   (requires CROFAI_API_KEY)
usvc_seller data populate
```

## CI workflows

| Workflow | Trigger | Purpose |
|---|---|---|
| `validate-data.yml` | every PR + push to main | schema compliance, file references, directory naming |
| `format-check.yml` | every PR + push to main | JSON/TOML formatting check |
| `upload-data.yml` | PR merged to main | push catalog to the UnitySVC backend, write back service-id overrides |
| `populate-services.yml` | daily at 02:00 UTC + manual | re-scrape CrofAI's model list and open a PR if anything changed |

## Required GitHub secrets

| Secret | Used by | Notes |
|---|---|---|
| `UNITYSVC_API_KEY` | upload-data | Seller API key from the UnitySVC console (mapped to `UNITYSVC_SELLER_API_KEY` at runtime). |
| `SERVICE_API_URL` | upload-data | UnitySVC API base URL. |
| `CROFAI_API_KEY` | populate-services | CrofAI API key — needed to call `https://ai.nahcrof.com/v2/models` to refresh the catalog. |

## Notes on `listing.override.json`

The first time a service is uploaded, the backend assigns it a UUID. The
upload workflow writes that UUID back into a sibling
`listing.override.json` and commits to `main`, which is how subsequent
populate runs match upstream model entries to existing listings instead of
creating duplicates. Don't edit these files by hand.
