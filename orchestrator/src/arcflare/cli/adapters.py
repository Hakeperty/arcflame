"""CLI tool integration adapters for OpenCode, Claude Code, Qwen Code."""

import logging
from typing import Dict, Optional

logger = logging.getLogger("arcflare.cli")


def generate_opencode_config(orchestrator_url: str) -> dict:
    """Generate an OpenCode provider configuration pointing at ArcFlare.

    Usage in opencode.json:
    {
        "provider": {
            "id": "arcflare",
            "config": { ... }
        }
    }
    """
    return {
        "id": "arcflare",
        "model": "arcflare/default",
        "api_key_env_var": "ARCFLARE_API_KEY",
        "urls": {
            "base_url": f"{orchestrator_url}/v1",
        },
    }


def generate_opencode_config_content(orchestrator_url: str) -> str:
    """Generate OPENCODE_CONFIG_CONTENT for ArcFlare proxy."""
    import json
    config = {
        "provider": {
            "id": "arcflare",
            "name": "ArcFlare Cluster",
            "model": "arcflare/default",
            "api_key_env_var": "ARCFLARE_API_KEY",
            "urls": {
                "base_url": f"{orchestrator_url}/v1",
            },
            "capabilities": {
                "streaming": True,
                "tools": True,
            },
        },
    }
    return json.dumps(config)


def get_claude_code_env(orchestrator_url: str) -> Dict[str, str]:
    """Get environment variables to point Claude Code at ArcFlare."""
    return {
        "ANTHROPIC_BASE_URL": orchestrator_url,
        "ANTHROPIC_API_KEY": "arcflare-dev-key",
    }


def get_qwen_code_env(orchestrator_url: str) -> Dict[str, str]:
    """Get environment variables to point Qwen Code at ArcFlare."""
    return {
        "OPENAI_BASE_URL": f"{orchestrator_url}/v1",
        "OPENAI_API_KEY": "arcflare-dev-key",
    }


def get_generic_env(orchestrator_url: str) -> Dict[str, str]:
    """Get environment variables for any OpenAI-compatible CLI tool."""
    return {
        "OPENAI_BASE_URL": f"{orchestrator_url}/v1",
        "OPENAI_API_KEY": "arcflare-dev-key",
    }


def print_setup_instructions(orchestrator_url: str = "http://localhost:8000"):
    """Print CLI tool setup instructions."""
    print(f"""
╔══════════════════════════════════════════════════════════╗
║              ArcFlare CLI Setup                          ║
╚══════════════════════════════════════════════════════════╝

Orchestrator URL: {orchestrator_url}

┌─────────────────────────────────────────────────────────┐
│ OpenCode                                                │
├─────────────────────────────────────────────────────────┤
│ Add to opencode.json:                                   │
│                                                         │
│   OPENCODE_CONFIG_CONTENT='{generate_opencode_config_content(orchestrator_url)}' \\
│     opencode
│
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Claude Code                                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ANTHROPIC_BASE_URL={orchestrator_url} \\
│   ANTHROPIC_API_KEY=arcflare-dev-key \\
│     claude
│
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Qwen Code / Any OpenAI-compatible CLI                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   OPENAI_BASE_URL={orchestrator_url}/v1 \\
│   OPENAI_API_KEY=arcflare-dev-key \\
│     qwen
│
└─────────────────────────────────────────────────────────┘
""")
