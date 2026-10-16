"""OpenAI Responses and Anthropic Messages adapters for fixed puzzle prompts."""
import base64
import io


def output_schema(parts):
    prompt = "\n".join(part["text"] for part in parts if "text" in part)
    strings = {"type": "array", "items": {"type": "string"}}
    if '"labels"' in prompt:
        properties = {"labels": strings}
    elif '"words"' in prompt:
        if "First identify one word for each image." in prompt:
            properties = {"words": strings, "concept": {"type": "string"}}
        else:
            properties = {"concept": {"type": "string"}, "words": strings}
    else:
        properties = {"word": {"type": "string"}, "concept": {"type": "string"}}
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def valid_answer(answer, parts):
    if not isinstance(answer, dict):
        return False
    properties = output_schema(parts)["properties"]
    if set(answer) != set(properties):
        return False
    for key, schema in properties.items():
        value = answer[key]
        if schema["type"] == "string":
            if not isinstance(value, str):
                return False
        elif not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            return False
        elif len(value) != (5 if key == "labels" else 3):
            return False
    return True


def request_body(parts, provider, model, max_output_tokens, reasoning_effort, anthropic_thinking):
    if provider not in {"openai", "anthropic"}:
        raise ValueError(f"Unknown hosted provider: {provider}")
    if max_output_tokens <= 0:
        raise ValueError("Hosted requests require a positive output-token cap")
    schema = output_schema(parts)
    if provider == "openai":
        for key, value in schema["properties"].items():
            if value["type"] == "array":
                count = 5 if key == "labels" else 3
                value.update(minItems=count, maxItems=count)
        content = []
        for part in parts:
            if part["type"] == "text":
                content.append({"type": "input_text", "text": part["text"]})
            else:
                content.append({"type": "input_image", "image_url": part["image_url"]["url"]})
        return {"model": model, "input": [{"role": "user", "content": content}],
                "reasoning": {"effort": reasoning_effort}, "max_output_tokens": max_output_tokens,
                "store": False, "text": {"format": {"type": "json_schema", "name": "puzzle_answer",
                                                     "strict": True, "schema": schema}}}
    content = []
    for part in parts:
        if part["type"] == "text":
            content.append(part)
        else:
            prefix, encoded = part["image_url"]["url"].split(",", 1)
            content.append({"type": "image", "source": {"type": "base64",
                            "media_type": prefix[5:].split(";")[0], "data": encoded}})
    config = {"format": {"type": "json_schema", "schema": schema}}
    if reasoning_effort != "default":
        if model.startswith("claude-haiku"):
            raise ValueError("Haiku 4.5 does not support effort; use --reasoning-effort default")
        config["effort"] = reasoning_effort
    if model.startswith("claude-opus-5-5") and anthropic_thinking != "adaptive":
        raise ValueError("Opus 5.5 requires adaptive thinking")
    return {"model": model, "max_tokens": max_output_tokens,
            "thinking": {"type": anthropic_thinking}, "output_config": config,
            "messages": [{"role": "user", "content": content}]}


def openai_image_bytes(data, mime):
    from PIL import Image
    with Image.open(io.BytesIO(data)) as photo:
        width, height = photo.size
        if ((width + 31) // 32) * ((height + 31) // 32) <= 30000:
            return data, mime, None
        original_size = photo.size
        # A 5500-pixel bounding box stays below 30,000 32-pixel patches.
        photo.thumbnail((5500, 5500), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        photo.convert("RGB").save(buffer, format="JPEG", quality=95)
        return buffer.getvalue(), "image/jpeg", {
            "original_size": list(original_size), "upload_size": list(photo.size),
            "reason": "openai_30000_patch_limit",
        }


def anthropic_image_bytes(data, mime):
    from PIL import Image
    # Stay within the documented high-resolution tier and base64 upload size.
    with Image.open(io.BytesIO(data)) as photo:
        if max(photo.size) <= 2576 and len(base64.b64encode(data)) < 10_000_000:
            return data, mime
        photo.thumbnail((2576, 2576), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        photo.convert("RGB").save(buffer, format="JPEG", quality=95)
        return buffer.getvalue(), "image/jpeg"


def decode_response(payload, provider):
    usage = dict(payload.get("usage") or {})
    usage["response_model"] = payload.get("model", "")
    usage["response_id"] = payload.get("id", "")
    if provider == "openai":
        usage["response_status"] = payload.get("status", "")
        usage["incomplete_details"] = payload.get("incomplete_details")
        blocks = [block for item in payload.get("output", []) if item.get("type") == "message"
                  for block in item.get("content", [])]
        text = "".join(block.get("text", "") for block in blocks if block.get("type") == "output_text")
    else:
        usage["stop_reason"] = payload.get("stop_reason", "")
        blocks = payload.get("content", [])
        text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    return text, usage
