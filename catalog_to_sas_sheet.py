#!/usr/bin/env python3
"""Utility for exporting catalog data to Google Sheets for SellerAmp SAS."""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import requests
from dotenv import load_dotenv
from gspread import Worksheet
import gspread


logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {"product name", "cost price"}
OPTIONAL_COLUMNS = {"asin", "ean", "sale price"}
AMAZON_SEARCH_ENDPOINT = "https://api.rainforestapi.com/request"
DEFAULT_SPREADSHEET_ID = "1Q0RmoCgONCwld0cpcSFWf-bWWEMdCnBaiPRuEdv38CA"
DEFAULT_WORKSHEET_TITLE = "Feuille 1"


@dataclass
class ExportConfig:
    spreadsheet_id: str
    worksheet_title: str
    min_roi_percent: Optional[float] = None
    min_profit_eur: Optional[float] = None
    enable_amazon_lookup: bool = False
    amazon_api_key: Optional[str] = None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export catalog products to the SellerAmp SAS Google Sheet",
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the catalog file (CSV or Excel).",
    )
    parser.add_argument(
        "--sheet-id",
        dest="spreadsheet_id",
        default=os.getenv("GOOGLE_SHEET_ID", DEFAULT_SPREADSHEET_ID),
        help="Override the Google Sheet ID (default: value from env or official sheet).",
    )
    parser.add_argument(
        "--worksheet",
        dest="worksheet_title",
        default=os.getenv("GOOGLE_WORKSHEET_TITLE", DEFAULT_WORKSHEET_TITLE),
        help="Override the Google worksheet title (default: value from env or 'Feuille 1').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process the catalog but do not push data to Google Sheets.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> ExportConfig:
    min_roi = _parse_float_env("MIN_ROI_PERCENT")
    min_profit = _parse_float_env("MIN_PROFIT_EUR")
    amazon_api_key = os.getenv("RAINFOREST_API_KEY")
    enable_lookup = bool(amazon_api_key) or os.getenv("ENABLE_AMAZON_LOOKUP", "").lower() in {"1", "true", "yes"}

    return ExportConfig(
        spreadsheet_id=args.spreadsheet_id,
        worksheet_title=args.worksheet_title,
        min_roi_percent=min_roi,
        min_profit_eur=min_profit,
        enable_amazon_lookup=enable_lookup,
        amazon_api_key=amazon_api_key,
    )


def _parse_float_env(name: str) -> Optional[float]:
    value = os.getenv(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        logger.warning("Environment variable %s should be a number, got %r", name, value)
        return None


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file {path} not found")

    if path.suffix.lower() in {".csv"}:
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file type. Provide a CSV or Excel file.")

    if df.empty:
        raise ValueError("The input catalog file is empty.")

    normalized_columns = {}
    for column in df.columns:
        normalized_columns[column] = column.strip().lower()

    df = df.rename(columns=normalized_columns)

    missing_required = REQUIRED_COLUMNS - set(df.columns)
    if missing_required:
        raise ValueError(
            "The catalog file must include columns: "
            + ", ".join(sorted(missing_required))
        )

    available_optional = OPTIONAL_COLUMNS & set(df.columns)
    if not available_optional:
        logger.warning(
            "Neither ASIN nor EAN column detected. Catalog rows without ASIN cannot be exported."
        )

    return df


def build_gspread_client() -> gspread.Client:
    credentials_raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_INFO")
    credentials_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_PATH")

    if credentials_raw:
        try:
            credentials_dict = json.loads(credentials_raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_INFO is not valid JSON") from exc
        return gspread.service_account_from_dict(credentials_dict)

    if credentials_path:
        cred_path = Path(credentials_path)
        if not cred_path.exists():
            raise FileNotFoundError(
                f"Service account JSON file defined in GOOGLE_SERVICE_ACCOUNT_JSON_PATH does not exist: {cred_path}"
            )
        return gspread.service_account(filename=str(cred_path))

    raise RuntimeError(
        "Google credentials not provided. Set GOOGLE_SERVICE_ACCOUNT_INFO or GOOGLE_SERVICE_ACCOUNT_JSON_PATH in the .env file."
    )


def get_worksheet(client: gspread.Client, config: ExportConfig) -> Worksheet:
    try:
        spreadsheet = client.open_by_key(config.spreadsheet_id)
    except gspread.SpreadsheetNotFound as exc:
        raise RuntimeError(
            "Unable to access the spreadsheet. Check GOOGLE_SHEET_ID permissions."
        ) from exc

    try:
        return spreadsheet.worksheet(config.worksheet_title)
    except gspread.WorksheetNotFound:
        logger.info(
            "Worksheet '%s' not found; creating it.", config.worksheet_title
        )
        worksheet = spreadsheet.add_worksheet(title=config.worksheet_title, rows=100, cols=8)
        header = [
            "Export Date (UTC)",
            "Product Name",
            "ASIN",
            "Amazon URL",
            "Cost Price",
            "Sale Price",
            "Profit",
            "Profit Margin",
        ]
        worksheet.insert_row(header, index=1)
        return worksheet


def lookup_asin(product_name: str, ean: Optional[str], config: ExportConfig) -> Optional[str]:
    if not config.enable_amazon_lookup:
        return None

    if not config.amazon_api_key:
        logger.debug("Amazon lookup enabled but no API key provided. Skipping lookup.")
        return None

    params = {
        "api_key": config.amazon_api_key,
        "type": "search",
        "amazon_domain": "amazon.fr",
        "search_term": ean or product_name,
        "page": 1,
    }

    try:
        response = requests.get(AMAZON_SEARCH_ENDPOINT, params=params, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Amazon lookup failed for %s (%s): %s", product_name, ean, exc)
        return None

    try:
        payload = response.json()
    except ValueError as exc:
        logger.warning("Invalid JSON response from Amazon lookup for %s: %s", product_name, exc)
        return None

    for item in payload.get("search_results", []) or []:
        asin = item.get("asin")
        if asin:
            logger.info("Lookup found ASIN %s for %s", asin, product_name)
            return asin

    logger.info("Amazon lookup returned no ASIN for %s", product_name)
    return None


def compute_profit_and_roi(cost_price: Optional[float], sale_price: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    if cost_price is None or cost_price <= 0 or sale_price is None:
        return None, None

    profit = sale_price - cost_price
    roi = (profit / cost_price) * 100
    return profit, roi


def format_currency(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def sanitize_numeric(value: object) -> Optional[float]:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.debug("Unable to convert %r to float", value)
        return None


def prepare_rows(df: pd.DataFrame, config: ExportConfig) -> tuple[List[List[str]], List[str], int]:
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    values: List[List[str]] = []
    missing_asin_products: List[str] = []
    filtered_out = 0
    has_sale_price_column = "sale price" in df.columns

    for _, row in df.iterrows():
        product_name = str(row.get("product name", "")).strip()
        if not product_name:
            logger.debug("Skipping row with empty product name: %s", row.to_dict())
            continue

        asin_raw = row.get("asin")
        asin = None
        if asin_raw is not None and not pd.isna(asin_raw):
            asin_candidate = str(asin_raw).strip()
            if asin_candidate:
                asin = asin_candidate

        ean_raw = row.get("ean")
        ean = None
        if ean_raw is not None and not pd.isna(ean_raw):
            ean_candidate = str(ean_raw).strip()
            if ean_candidate:
                ean = ean_candidate

        if not asin:
            asin = lookup_asin(product_name, ean, config)

        if not asin:
            missing_asin_products.append(product_name)
            continue

        cost_price = sanitize_numeric(row.get("cost price"))
        if cost_price is None:
            logger.warning("Skipping %s due to invalid cost price: %r", product_name, row.get("cost price"))
            continue

        sale_price = sanitize_numeric(row.get("sale price")) if has_sale_price_column else None

        profit, roi = compute_profit_and_roi(cost_price, sale_price)

        if config.min_roi_percent is not None and roi is not None and roi < config.min_roi_percent:
            filtered_out += 1
            continue

        if config.min_profit_eur is not None and profit is not None and profit < config.min_profit_eur:
            filtered_out += 1
            continue

        values.append(
            [
                date_str,
                product_name,
                asin,
                f"https://www.amazon.fr/dp/{asin}",
                format_currency(cost_price),
                format_currency(sale_price),
                "",
                "",
            ]
        )

    return values, missing_asin_products, filtered_out


def append_rows(worksheet: Worksheet, rows: Iterable[Sequence[str]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        logger.info("No rows to append to Google Sheets.")
        return
    worksheet.append_rows(rows_list, value_input_option="USER_ENTERED")


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    setup_logging(args.log_level)

    try:
        df = load_catalog(args.input_file)
    except Exception as exc:
        logger.error("Unable to load catalog: %s", exc)
        return 1

    config = load_config(args)
    rows, missing_asin_products, filtered_out = prepare_rows(df, config)

    if missing_asin_products:
        logger.warning("%d product(s) skipped due to missing ASIN: %s", len(missing_asin_products), "; ".join(missing_asin_products))

    if filtered_out:
        logger.info("%d product(s) skipped by ROI/profit filters.", filtered_out)

    if not rows:
        logger.info("No rows prepared for export. Nothing to send to Google Sheets.")
        return 0

    if args.dry_run:
        logger.info("Dry run enabled. Prepared %d row(s) but did not update Google Sheets.", len(rows))
        return 0

    try:
        client = build_gspread_client()
        worksheet = get_worksheet(client, config)
        append_rows(worksheet, rows)
    except Exception as exc:
        logger.error("Failed to update Google Sheets: %s", exc)
        return 1

    print(f"Added {len(rows)} product(s) to Google Sheets.")
    if missing_asin_products:
        print(
            "Products without ASIN (not exported): " + "; ".join(missing_asin_products)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
