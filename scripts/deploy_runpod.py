#!/usr/bin/env python3
"""Deploy TTS engines to RunPod serverless."""

import os
import sys

try:
    import requests
except ImportError:
    os.system(f"{sys.executable} -m pip install requests -q")
    import requests


RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"

# Engine configurations
ENGINES = {
    "kokoro": {
        "image": "ghcr.io/gbstox/tts-kokoro:latest",
        "template_name": "tts-kokoro-template",
    },
    "cosyvoice": {
        "image": "ghcr.io/gbstox/tts-cosyvoice:latest",
        "template_name": "tts-cosyvoice-template",
    },
    "fishspeech": {
        "image": "ghcr.io/gbstox/tts-fishspeech:latest",
        "template_name": "tts-fishspeech-template",
    },
    "qwen3tts": {
        "image": "ghcr.io/gbstox/tts-qwen3tts:latest",
        "template_name": "TTS-qwen3tts",
    },
}


def graphql_query(api_key: str, query: str, variables: dict = None) -> dict:
    """Execute GraphQL query."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    response = requests.post(RUNPOD_GRAPHQL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def get_templates_and_endpoints(api_key: str) -> tuple[dict, dict]:
    """Get existing templates and endpoints."""
    query = """
    {
        myself {
            podTemplates { id name imageName }
            endpoints { id name templateId }
        }
    }
    """
    result = graphql_query(api_key, query)
    data = result.get("data", {}).get("myself", {})
    
    templates = {t["name"]: t for t in data.get("podTemplates", [])}
    endpoints = {e["name"]: e for e in data.get("endpoints", [])}
    
    return templates, endpoints


def update_template_image(api_key: str, template_id: str, new_image: str) -> dict:
    """Update template with new image."""
    query = """
    mutation SavePodTemplate($input: PodTemplateInput!) {
        savePodTemplate(input: $input) {
            id
            name
            imageName
        }
    }
    """
    variables = {
        "input": {
            "id": template_id,
            "imageName": new_image,
        }
    }
    return graphql_query(api_key, query, variables)


def create_endpoint(api_key: str, name: str, template_id: str) -> dict:
    """Create a serverless endpoint."""
    query = """
    mutation SaveEndpoint($input: EndpointInput!) {
        saveEndpoint(input: $input) {
            id
            name
        }
    }
    """
    variables = {
        "input": {
            "name": name,
            "templateId": template_id,
            "gpuIds": "AMPERE_16",
            "workersMin": 0,
            "workersMax": 3,
            "idleTimeout": 5,
            "flashBoot": True,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 4,
        }
    }
    return graphql_query(api_key, query, variables)


def main():
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("Error: RUNPOD_API_KEY not set")
        sys.exit(1)
    
    print("RunPod TTS Deployment")
    print("=" * 50)
    
    # Get existing templates and endpoints
    print("\nFetching existing resources...")
    templates, endpoints = get_templates_and_endpoints(api_key)
    print(f"  Found {len(templates)} templates, {len(endpoints)} endpoints")
    
    # Process each engine
    results = []
    for engine_id, config in ENGINES.items():
        print(f"\n--- {engine_id.upper()} ---")
        template_name = config["template_name"]
        new_image = config["image"]
        
        if template_name in templates:
            template = templates[template_name]
            current_image = template.get("imageName", "")
            
            if current_image == new_image:
                print(f"  Template: Already using {new_image}")
            else:
                print(f"  Template: Updating to {new_image}")
                try:
                    update_template_image(api_key, template["id"], new_image)
                    print(f"  ✓ Template updated")
                except Exception as e:
                    print(f"  ✗ Failed: {e}")
            
            # Check endpoint
            endpoint_name = f"tts-{engine_id}"
            if endpoint_name in endpoints:
                print(f"  Endpoint: Already exists ({endpoints[endpoint_name]['id']})")
            else:
                print(f"  Endpoint: Creating...")
                try:
                    result = create_endpoint(api_key, endpoint_name, template["id"])
                    if "errors" in result:
                        print(f"  ✗ Failed: {result['errors']}")
                    else:
                        ep = result.get("data", {}).get("saveEndpoint", {})
                        print(f"  ✓ Endpoint created: {ep.get('id')}")
                except Exception as e:
                    print(f"  ✗ Failed: {e}")
        else:
            print(f"  ✗ Template '{template_name}' not found!")
    
    print("\n" + "=" * 50)
    print("Deployment complete!")
    print("\nYour RunPod endpoints:")
    
    # Refresh endpoints
    _, endpoints = get_templates_and_endpoints(api_key)
    for engine_id in ENGINES.keys():
        endpoint_name = f"tts-{engine_id}"
        if endpoint_name in endpoints:
            ep_id = endpoints[endpoint_name]["id"]
            print(f"  {engine_id}: https://api.runpod.ai/v2/{ep_id}/runsync")


if __name__ == "__main__":
    main()
