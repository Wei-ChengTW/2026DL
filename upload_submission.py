"""Upload a completed RSNA competition submission using a local Kaggle token."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import requests

from rsna_detector import DEFAULT_OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_OUTPUT_DIR / "submission.csv")
    parser.add_argument("--competition", default="rsna-pneumonia-detection-challenge")
    parser.add_argument("--message", default="Faster R-CNN ResNet50-FPN")
    parser.add_argument("--token-file", type=Path, default=Path.home() / ".kaggle" / "access_token")
    parser.add_argument("--poll-seconds", type=int, default=900)
    parser.add_argument("--poll-interval", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.file.is_file():
        raise SystemExit(f"Submission file not found: {args.file}")
    if args.poll_seconds < 0 or args.poll_interval < 1:
        raise SystemExit("Polling timeout must be non-negative and interval must be positive.")

    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token and args.token_file.is_file():
        token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit(
            f"No Kaggle API token found in KAGGLE_API_TOKEN or {args.token_file}."
        )
    # Keep the token in process memory; never print it or write it into the project.
    os.environ["KAGGLE_API_TOKEN"] = token

    from kagglesdk.competitions.types.competition_api_service import (
        ApiCreateSubmissionRequest,
        ApiGetSubmissionRequest,
        ApiStartSubmissionUploadRequest,
        SubmissionStatus,
    )
    from kagglesdk.kaggle_client import KaggleClient
    from kagglesdk.kaggle_env import KaggleEnv

    start_request = ApiStartSubmissionUploadRequest()
    start_request.competition_name = args.competition
    start_request.file_name = args.file.name
    start_request.content_length = args.file.stat().st_size
    start_request.last_modified_epoch_seconds = int(args.file.stat().st_mtime)

    with KaggleClient(env=KaggleEnv.PROD) as kaggle:
        competition_api = kaggle.competitions.competition_api_client
        upload = competition_api.start_submission_upload(start_request)
        with args.file.open("rb") as handle:
            upload_response = requests.put(upload.create_url, data=handle, timeout=300)
        if upload_response.status_code not in (200, 201):
            raise SystemExit(
                f"Kaggle file upload failed with HTTP {upload_response.status_code}."
            )

        submit_request = ApiCreateSubmissionRequest()
        submit_request.competition_name = args.competition
        submit_request.blob_file_tokens = upload.token
        submit_request.submission_description = args.message
        submitted = competition_api.create_submission(submit_request)
        print(f"Submission accepted by Kaggle; reference={submitted.ref}.", flush=True)
        if submitted.message:
            print(submitted.message, flush=True)

        deadline = time.monotonic() + args.poll_seconds
        while True:
            status_request = ApiGetSubmissionRequest()
            status_request.ref = submitted.ref
            result = competition_api.get_submission(status_request)
            status = result.status
            status_name = getattr(status, "name", str(status))
            if status == SubmissionStatus.COMPLETE:
                print(f"Kaggle public score: {result.public_score}", flush=True)
                return
            if status == SubmissionStatus.ERROR:
                details = result.error_description.replace(token, "[redacted]")
                raise SystemExit(f"Kaggle marked the submission as ERROR: {details}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(
                    f"Kaggle is still scoring submission {submitted.ref} "
                    f"(status={status_name}); poll again to retrieve its score.",
                    flush=True,
                )
                return
            print(f"Kaggle scoring status: {status_name}; waiting for the public score.", flush=True)
            time.sleep(min(args.poll_interval, remaining))


if __name__ == "__main__":
    main()
