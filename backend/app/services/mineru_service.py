import json
import subprocess

from app.core.config import settings
from app.core.exceptions import DocumentExtractionException


class MineruService:

    def parse_document(self, file_path: str) -> dict:
        command = [
            settings.MINERU_COMMAND,
            "parse",
            file_path,
            "--tier",
            settings.MINERU_TIER,
            "--json",
        ]

        if settings.MINERU_REMOTE:
            command.append("--remote")

        try:
            response = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=settings.MINERU_TIMEOUT,
            )

        except subprocess.TimeoutExpired as exc:
            raise DocumentExtractionException(
                "MinerU parsing timed out."
            ) from exc

        except FileNotFoundError as exc:
            raise DocumentExtractionException(
                f"MinerU executable not found: "
                f"{settings.MINERU_COMMAND}"
            ) from exc

        except Exception as exc:
            raise DocumentExtractionException(
                f"Failed to execute MinerU: {exc}"
            ) from exc

        if response.returncode != 0:
            raise DocumentExtractionException(
                f"MinerU parsing failed: {response.stderr}"
            )

        if not response.stdout:
            raise DocumentExtractionException(
                "MinerU returned no output."
            )

        try:
            result = json.loads(response.stdout)

        except json.JSONDecodeError as exc:
            raise DocumentExtractionException(
                "MinerU returned invalid JSON."
            ) from exc

        parse_info = result.get("parse", {})
        content = result.get("content", {})

        markdown = content.get("content", "")

        if not markdown:
            raise DocumentExtractionException(
                "MinerU returned no document content."
            )

        return {
            "sha256": parse_info.get("sha256"),
            "short_id": parse_info.get("short_id"),
            "tier": parse_info.get("tier"),
            "page_range": parse_info.get("page_range"),
            "markdown": markdown,
        }


mineru_service = MineruService()